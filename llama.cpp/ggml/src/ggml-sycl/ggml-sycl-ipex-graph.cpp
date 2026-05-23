// Layer 3: Graph-level fusion detection
// Detects QKV and MLP patterns in the GGML computation graph and replaces
// 3 separate mul_mat ops with 1 fused kernel call.
//
// This operates at the ggml_cgraph level — BEFORE backend execution.
// IPEX-LLM does this in libllama.so (build_bigdl_qkv_fusion, etc.)

#include "ggml-sycl-ipex.h"
#include <ggml-impl.h>
#include <cstring>
#include <cstdio>

namespace ipex_fusion {

// ============================================================
// Pattern detection helpers
// ============================================================

// Check if two tensors share the same src1 (same activation input)
static bool same_src1(const ggml_tensor *a, const ggml_tensor *b) {
    if (!a || !b) return false;
    if (a->n_src < 2 || b->n_src < 2) return false;
    return a->src[1] == b->src[1];
}

// Check if a tensor is a MUL_MAT op with quantized weights
static bool is_quantized_mul_mat(const ggml_tensor *t) {
    if (!t || t->op != GGML_OP_MUL_MAT) return false;
    if (t->n_src < 1) return false;
    return ggml_is_quantized(t->src[0]->type);
}

// Check if tensor dimensions match for QKV fusion
// Q/K/V projections: each src0 should have the same shape (n_embd × n_embd_head)
// unless using GQA where K/V have fewer heads
static bool is_qkv_compatible(const ggml_tensor *q_proj,
                               const ggml_tensor *k_proj,
                               const ggml_tensor *v_proj)
{
    if (!q_proj || !k_proj || !v_proj) return false;
    // src0 is the weight matrix: ne0 = n_embd, ne1 = n_head * d_head
    // For GQA: K and V have ne1 = n_head_kv * d_head (smaller than Q)
    int64_t n_embd    = q_proj->ne[0];
    int64_t q_heads   = q_proj->ne[1];  // n_head_total * d_head
    int64_t kv_heads  = k_proj->ne[1];  // n_head_kv * d_head (GQA)

    // K and V must have same dimensions as each other
    if (k_proj->ne[0] != v_proj->ne[0]) return false;
    if (k_proj->ne[1] != v_proj->ne[1]) return false;

    // All must share the same n_embd dimension
    if (k_proj->ne[0] != n_embd) return false;

    // Q may be larger (GQA) or same (MHA)
    (void)q_heads;
    (void)kv_heads;

    return true;
}

// ============================================================
// QKV Pattern Detection
// ============================================================

// In a standard attention block, Q/K/V projections appear as 3 consecutive
// MUL_MAT ops sharing the same src1 (hidden states / activations).
//
// Graph pattern:
//   q = mul_mat(W_q, hidden)   ← src0=W_q (Q4_0), src1=hidden
//   k = mul_mat(W_k, hidden)   ← src0=W_k (Q4_0), src1=same hidden
//   v = mul_mat(W_v, hidden)   ← src0=W_v (Q4_0), src1=same hidden
//
// Fusion replaces these 3 with:
//   [q,k,v] = qkv_fusion(W_q, W_k, W_v, hidden)
//
// This saves:
//   - 2 kernel launches
//   - 2 reads of the hidden states from memory
//   - 3 sets of quantization parameters computed on GPU

QKVPattern detect_qkv_pattern(const ggml_tensor *q_tensor,
                               const ggml_tensor *k_tensor,
                               const ggml_tensor *v_tensor)
{
    QKVPattern p;

    if (!is_quantized_mul_mat(q_tensor)) return p;
    if (!is_quantized_mul_mat(k_tensor)) return p;
    if (!is_quantized_mul_mat(v_tensor)) return p;

    // All three must share the same src1 (activations)
    if (!same_src1(q_tensor, k_tensor)) return p;
    if (!same_src1(q_tensor, v_tensor)) return p;

    // src0 weights must be compatible
    if (!is_qkv_compatible(q_tensor->src[0], k_tensor->src[0], v_tensor->src[0]))
        return p;

    // Only fuse if all use the same quant type (currently Q4_0 for the fusion kernel)
    // TODO: extend to Q4_K when we have that kernel
    ggml_type qt = q_tensor->src[0]->type;
    if (k_tensor->src[0]->type != qt || v_tensor->src[0]->type != qt)
        return p;

    // Currently only Q4_0 is supported by the IPEX QKV fusion SPIR-V kernel
    if (qt != GGML_TYPE_Q4_0)
        return p;

    p.q_proj = q_tensor->src[0];
    p.k_proj = k_tensor->src[0];
    p.v_proj = v_tensor->src[0];
    p.input  = q_tensor->src[1];
    p.valid  = true;

    return p;
}

// ============================================================
// MLP Pattern Detection
// ============================================================

// In a standard FFN/MLP block:
//   gate = mul_mat(W_gate, hidden)   ← src0=W_gate, src1=hidden
//   up   = mul_mat(W_up,   hidden)   ← src0=W_up,   src1=same hidden
//   (activation function: SiLU(gate) * up)
//   down = mul_mat(W_down, result)   ← separate, can't fuse with gate/up
//
// We can fuse gate+up into:
//   [gate, up] = mlp_fusion(W_gate, W_up, hidden)
//
// This saves:
//   - 1 kernel launch
//   - 1 read of hidden states

MLPPattern detect_mlp_pattern(const ggml_tensor *gate_tensor,
                               const ggml_tensor *up_tensor,
                               const ggml_tensor *down_tensor)
{
    MLPPattern p;
    (void)down_tensor;  // down uses different src1, can't fuse

    if (!is_quantized_mul_mat(gate_tensor)) return p;
    if (!is_quantized_mul_mat(up_tensor))   return p;

    // Gate and up must share the same src1
    if (!same_src1(gate_tensor, up_tensor)) return p;

    // Weights must have same dimensions (n_embd × n_ff)
    if (gate_tensor->src[0]->ne[0] != up_tensor->src[0]->ne[0]) return p;
    if (gate_tensor->src[0]->ne[1] != up_tensor->src[0]->ne[1]) return p;

    // Same quant type
    ggml_type gt = gate_tensor->src[0]->type;
    if (up_tensor->src[0]->type != gt) return p;

    // MLP fusion kernel currently supports Q4_0 and Q4_K
    if (gt != GGML_TYPE_Q4_0 && gt != GGML_TYPE_Q4_K) return p;

    p.gate_proj = gate_tensor->src[0];
    p.up_proj   = up_tensor->src[0];
    p.input     = gate_tensor->src[1];
    p.valid     = true;

    return p;
}

// ============================================================
// Graph traversal: scan all nodes for fusion opportunities
// ============================================================

struct FusionPlan {
    std::vector<QKVPattern> qkv_ops;   // QKV patterns found
    std::vector<MLPPattern> mlp_ops;   // MLP patterns found

    int potential_savings() const {
        // Each QKV fusion saves 2 kernel launches + memory reads
        // Each MLP fusion saves 1 kernel launch + memory read
        return qkv_ops.size() * 2 + mlp_ops.size() * 1;
    }
};

FusionPlan analyze_graph(ggml_cgraph *graph) {
    FusionPlan plan;

    // Walk through the graph nodes looking for consecutive MUL_MAT ops
    // that share the same src1 (activation input)
    for (int i = 0; i < graph->n_nodes - 2; i++) {
        ggml_tensor *a = graph->nodes[i];
        ggml_tensor *b = graph->nodes[i + 1];
        ggml_tensor *c = graph->nodes[i + 2];

        if (!a || !b || !c) continue;
        if (a->op != GGML_OP_MUL_MAT || b->op != GGML_OP_MUL_MAT ||
            c->op != GGML_OP_MUL_MAT)
            continue;

        // Try QKV pattern first (3 consecutive matmuls sharing src1)
        QKVPattern qkv = detect_qkv_pattern(a, b, c);
        if (qkv.valid) {
            plan.qkv_ops.push_back(qkv);
            i += 2;  // skip the fused ops
            continue;
        }

        // Try MLP pattern (gate + up sharing src1, down is next)
        MLPPattern mlp = detect_mlp_pattern(a, b, c);
        if (mlp.valid) {
            plan.mlp_ops.push_back(mlp);
            i += 1;  // gate+up fused, down stays separate
            continue;
        }
    }

    if (!plan.qkv_ops.empty() || !plan.mlp_ops.empty()) {
        fprintf(stderr, "[IPEX-graph] Found %zu QKV + %zu MLP fusion opportunities"
                " (saves %d kernel launches)\n",
                plan.qkv_ops.size(), plan.mlp_ops.size(),
                plan.potential_savings());
    }

    return plan;
}

// ============================================================
// Graph rewriting: apply fusion plan
// ============================================================

// For now, this is a placeholder. Full graph rewriting requires:
// 1. Creating new fused tensor nodes
// 2. Reconnecting edges (consumers of q/k/v now read from fused output)
// 3. Removing the original 3 separate mul_mat nodes
//
// This is complex because downstream ops (rope, attention, etc.) expect
// separate q/k/v tensors. The fused kernel output needs to be "unpacked"
// into separate tensors via view/slice ops, OR we need to modify the
// attention implementation to read from a fused QKV buffer.

bool apply_fusion_plan(ggml_cgraph *graph, const FusionPlan &plan) {
    (void)graph;
    if (plan.qkv_ops.empty() && plan.mlp_ops.empty())
        return false;

    fprintf(stderr, "[IPEX-graph] Fusion plan: %zu QKV + %zu MLP ops\n"
            "  (graph rewriting not yet implemented — this is the long-term goal)\n",
            plan.qkv_ops.size(), plan.mlp_ops.size());

    // TODO: Implement graph rewriting
    // Approach:
    //   1. For each QKV pattern, create a fused MUL_MAT node
    //      with 3 src0 (W_q, W_k, W_v) and 1 src1 (hidden)
    //   2. Set op = GGML_OP_MUL_MAT (custom fusion variant via op_params)
    //   3. Replace references in downstream nodes
    //   4. Add view/slice nodes to unpack q, k, v for attention

    return false;
}

} // namespace ipex_fusion
