"""
实训4: Transformer自注意力机制手动实现

从零实现缩放点积注意力: Attention(Q,K,V) = softmax(QK^T/√d_k) V
"""

import numpy as np


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力
    
    Args:
        Q: Query矩阵 (seq_len, d_k)
        K: Key矩阵   (seq_len, d_k)
        V: Value矩阵 (seq_len, d_v)
        mask: 可选的注意力掩码
    
    Returns:
        output: 注意力输出 (seq_len, d_v)
        weights: 注意力权重 (seq_len, seq_len)
    """
    d_k = Q.shape[-1]

    # Step 1: 计算相似度分数 Q @ K^T
    scores = Q @ K.T  # (seq_len, seq_len)

    # Step 2: 缩放
    scores = scores / np.sqrt(d_k)

    # Step 3: 掩码 (如果有)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # Step 4: Softmax 归一化
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)

    # Step 5: 加权求和
    output = weights @ V

    return output, weights


def single_head_self_attention(X, W_Q, W_K, W_V):
    """单头自注意力: X → Q,K,V → Attention → Output"""
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    return scaled_dot_product_attention(Q, K, V)


def multi_head_attention_demo(X, num_heads=2):
    """多头注意力演示"""
    seq_len, d_model = X.shape
    d_k = d_model // num_heads  # 每个头的维度

    # 模拟多头: 将 d_model 拆分为 num_heads 个 d_k 维的子空间
    head_outputs = []
    head_weights = []

    for h in range(num_heads):
        # 每个头有自己的 Q,K,V 权重 (简化: 随机初始化)
        W_Q = np.random.randn(d_model, d_k) * 0.1
        W_K = np.random.randn(d_model, d_k) * 0.1
        W_V = np.random.randn(d_model, d_k) * 0.1

        Q_h = X @ W_Q
        K_h = X @ W_K
        V_h = X @ W_V

        out_h, w_h = scaled_dot_product_attention(Q_h, K_h, V_h)
        head_outputs.append(out_h)
        head_weights.append(w_h)

    # 拼接所有头
    multihead_out = np.concatenate(head_outputs, axis=-1)

    # 最终线性投影 (简化: 单位矩阵)
    W_O = np.eye(d_model)
    final_output = multihead_out @ W_O

    return final_output, head_weights


def demo():
    """完整演示"""
    np.random.seed(42)
    np.set_printoptions(precision=3, suppress=True)

    print("=" * 60)
    print("实训4: Transformer自注意力机制手动实现 (NumPy)")
    print("=" * 60)

    # ===== 参数设置 =====
    seq_len = 4   # 序列长度 (模拟4个词)
    d_model = 6   # 嵌入维度

    # 输入序列 (模拟词嵌入)
    X = np.random.randn(seq_len, d_model) * 0.5
    print(f"\n📥 输入序列 X ({seq_len}词 × {d_model}维):")
    print(X)

    # ===== 1. 单头自注意力 =====
    print(f"\n{'─' * 50}")
    print("1️⃣  单头自注意力")
    print(f"{'─' * 50}")

    W_Q = np.random.randn(d_model, d_model) * 0.1
    W_K = np.random.randn(d_model, d_model) * 0.1
    W_V = np.random.randn(d_model, d_model) * 0.1

    output, attn_weights = single_head_self_attention(X, W_Q, W_K, W_V)

    print(f"\n🔍 注意力权重矩阵 ({seq_len}×{seq_len}):")
    print("    (第i行=第i个词对每个词的关注度, 行和为1)")
    print(attn_weights)

    # 解读注意力
    print(f"\n📊 注意力解读:")
    for i in range(seq_len):
        most_attended = np.argmax(attn_weights[i])
        print(f"  词{i} 最关注 词{most_attended} "
              f"(权重={attn_weights[i, most_attended]:.3f})")

    print(f"\n📤 自注意力输出 ({seq_len}×{d_model}):")
    print(output)

    # ===== 2. 多头注意力 =====
    print(f"\n{'─' * 50}")
    print("2️⃣  多头注意力 (num_heads=2)")
    print(f"{'─' * 50}")

    final_out, head_weights = multi_head_attention_demo(X, num_heads=2)

    print(f"\n🔍 头1注意力权重:")
    print(head_weights[0])
    print(f"\n🔍 头2注意力权重:")
    print(head_weights[1])
    print(f"\n📤 多头拼接输出 ({seq_len}×{d_model}):")
    print(final_out)

    # ===== 3. 公式讲解 =====
    print(f"\n{'─' * 50}")
    print("3️⃣  核心公式回顾")
    print(f"{'─' * 50}")
    print("""
    Attention(Q, K, V) = softmax(─── QK^T ───) V
                                  √d_k

    步骤:
    ① Q @ K^T     → 计算Query与每个Key的相似度
    ② ÷ √d_k      → 缩放防止梯度消失
    ③ softmax()   → 归一化为概率分布
    ④ @ V         → 加权聚合Value

    多头: 并行计算多个Attention, 拼接后投影
    MultiHead = Concat(head_1, ..., head_h) @ W_O
    """)

    # ===== 4. 尝试可视化 (文本版) =====
    print(f"{'─' * 50}")
    print("4️⃣  注意力热力图 (文本版)")
    print(f"{'─' * 50}")

    def text_heatmap(matrix):
        """文本版热力图"""
        chars = " ░▒▓█"
        for row in matrix:
            line = ""
            for val in row:
                idx = min(int(val * len(chars)), len(chars) - 1)
                line += chars[idx] * 3
            print(f"  {line}  {row[0]:.2f}→{row[-1]:.2f}")

    print("\n单头注意力热力图:")
    text_heatmap(attn_weights)

    print(f"\n{'=' * 60}")
    print("实训4 完成!")
    print("理解: Q查询 '我想知道什么', K键 '我能提供什么', V值 '实际内容'")
    print("=" * 60)

    # ===== 5. 数学验证 =====
    print(f"\n🧮 数学验证: 手动计算 vs NumPy")
    # 手动验证 softmax
    scores = np.array([[2.0, 1.0, 0.1, 3.0]])
    manual_softmax = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    print(f"  scores: {scores}")
    print(f"  softmax: {manual_softmax}")
    print(f"  验证: 和={manual_softmax.sum():.1f} (应为1.0)")


def main():
    try:
        import matplotlib
        matplotlib.use('Agg')  # 非交互后端，避免弹窗阻塞
        import matplotlib.pyplot as plt

        # 如果有 matplotlib，生成真实热力图
        np.random.seed(42)
        seq_len, d_model = 4, 6
        X = np.random.randn(seq_len, d_model) * 0.5
        W_Q = np.random.randn(d_model, d_model) * 0.1
        W_K = np.random.randn(d_model, d_model) * 0.1
        W_V = np.random.randn(d_model, d_model) * 0.1

        _, attn_weights = single_head_self_attention(X, W_Q, W_K, W_V)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # 热力图
        im = axes[0].imshow(attn_weights, cmap='Blues', vmin=0, vmax=1)
        axes[0].set_title("Self-Attention Weight Matrix")
        axes[0].set_xlabel("Key Position")
        axes[0].set_ylabel("Query Position")
        plt.colorbar(im, ax=axes[0])
        for i in range(seq_len):
            for j in range(seq_len):
                axes[0].text(j, i, f"{attn_weights[i, j]:.2f}",
                             ha='center', va='center',
                             color='black' if attn_weights[i, j] < 0.6 else 'white')

        # 柱状图: 每个Query对Key的关注分布
        x = np.arange(seq_len)
        width = 0.2
        for i in range(seq_len):
            axes[1].bar(x + i * width, attn_weights[i], width,
                       label=f'Query {i}')
        axes[1].set_title("Attention Distribution per Query")
        axes[1].set_xlabel("Key Position")
        axes[1].set_ylabel("Weight")
        axes[1].set_xticks(x + width * 1.5)
        axes[1].set_xticklabels([f'Key {j}' for j in range(seq_len)])
        axes[1].legend()

        plt.tight_layout()
        save_path = "attention_heatmap.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"\n📊 热力图已保存: {save_path}")

    except ImportError:
        pass

    demo()


if __name__ == "__main__":
    main()
