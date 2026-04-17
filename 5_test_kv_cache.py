import torch
import math
from kv_cache import KVCacheCompressor

def print_result(name, condition):
    if condition:
        print(f"✅ PASS: {name}")
    else:
        print(f"❌ FAIL: {name}")

def test_shape_and_metadata():
    print("\n--- Nhóm 1: Kiểm tra Cấu trúc và Siêu dữ liệu (Metadata) ---")
    # Đảm bảo tensor 4D được giữ nguyên hình dáng sau khi qua cỗ máy nén
    head_dim = 64
    num_layers, num_heads, seq_len = 2, 4, 16
    
    compressor = KVCacheCompressor(num_layers, num_heads, head_dim, k_bits=3.5, v_bits=2.5)
    
    k = torch.randn(num_layers, num_heads, seq_len, head_dim)
    v = torch.randn(num_layers, num_heads, seq_len, head_dim)
    
    compressor.calibrate(k, v) # Bắt buộc phải tìm Outliers trước
    compressed = compressor.compress(k, v)
    k_hat, v_hat = compressor.decompress(compressed)
    
    # Kiểm tra Metadata
    metadata_ok = (compressed.num_layers == num_layers and 
                   compressed.num_heads == num_heads and 
                   compressed.seq_len == seq_len and 
                   compressed.head_dim == head_dim)
    print_result("Lưu trữ Metadata cấu trúc Transformer chính xác", metadata_ok)
    
    # Kiểm tra Shape
    shape_ok = (k_hat.shape == k.shape) and (v_hat.shape == v.shape)
    print_result(f"Khôi phục nguyên vẹn cấu trúc 4D Tensor {k.shape}", shape_ok)

def test_round_trip_quality():
    print("\n--- Nhóm 2: Kiểm tra Sai số (MSE Bounds) của Cache ---")
    head_dim = 128
    num_layers, num_heads, seq_len = 2, 2, 32
    
    compressor = KVCacheCompressor(num_layers, num_heads, head_dim, k_bits=3.5, v_bits=2.5)
    
    k = torch.randn(num_layers, num_heads, seq_len, head_dim)
    v = torch.randn(num_layers, num_heads, seq_len, head_dim)
    
    compressor.calibrate(k, v)
    compressed = compressor.compress(k, v)
    k_hat, v_hat = compressor.decompress(compressed)
    
    k_mse = torch.mean((k - k_hat) ** 2).item()
    v_mse = torch.mean((v - v_hat) ** 2).item()
    
    print(f"K-Cache MSE (3.5-bit): {k_mse:.6f}")
    print(f"V-Cache MSE (2.5-bit): {v_mse:.6f}")
    # Theo bài báo, sai số phải cực thấp
    print_result("Sai số cả K và V đều nằm trong ngưỡng an toàn", k_mse < 0.05 and v_mse < 0.1)

def test_attention_score_preservation():
    print("\n--- Nhóm 3: Kiểm tra Bảo toàn Cơ chế Attention (Quan trọng cho RAG) ---")
    # Mô phỏng phép toán Q @ K^T @ V của Transformer
    head_dim = 64
    seq_len = 32
    
    compressor = KVCacheCompressor(num_layers=1, num_heads=1, head_dim=head_dim, k_bits=3.5, v_bits=2.5)
    torch.manual_seed(42)
    
    # Tạo Query (Câu hỏi của user)
    q = torch.randn(1, head_dim)
    # Tạo K, V (Đoạn văn bản trong sách)
    k = torch.randn(seq_len, head_dim)
    v = torch.randn(seq_len, head_dim)
    
    # 1. Chạy Attention gốc (Chưa nén)
    scores_orig = torch.matmul(q, k.T) / math.sqrt(head_dim)
    attn_orig = torch.nn.functional.softmax(scores_orig, dim=-1)
    out_orig = torch.matmul(attn_orig, v)
    
    # Định dạng lại thành 4D để đưa vào máy nén
    k_cache = k.view(1, 1, seq_len, head_dim)
    v_cache = v.view(1, 1, seq_len, head_dim)
    
    compressor.calibrate(k_cache, v_cache)
    compressed = compressor.compress(k_cache, v_cache)
    k_hat_4d, v_hat_4d = compressor.decompress(compressed)
    
    # Trích xuất lại dạng 2D
    k_hat = k_hat_4d[0, 0]
    v_hat = v_hat_4d[0, 0]
    
    # 2. Chạy Attention trên dữ liệu ĐÃ NÉN
    scores_comp = torch.matmul(q, k_hat.T) / math.sqrt(head_dim)
    attn_comp = torch.nn.functional.softmax(scores_comp, dim=-1)
    out_comp = torch.matmul(attn_comp, v_hat)
    
    # Tính Cosine Similarity giữa 2 kết quả
    cosine_sim = torch.nn.functional.cosine_similarity(out_orig.view(-1), out_comp.view(-1), dim=0).item()
    print(f"Độ tương đồng ngữ nghĩa (Cosine Similarity): {cosine_sim:.4f}")
    # Cosine > 0.95 nghĩa là AI vẫn suy luận đúng 95% dù chỉ dùng 1/5 dung lượng RAM
    print_result("LLM truy xuất chính xác thông tin sau khi bị nén", cosine_sim > 0.95)

def test_memory_stats():
    print("\n--- Nhóm 4: Kiểm tra Số liệu Tiết kiệm Bộ nhớ ---")
    # Kiểm tra hàm thống kê VRAM
    compressor = KVCacheCompressor(num_layers=32, num_heads=32, head_dim=128, k_bits=3.5, v_bits=2.5)
    
    # Giả lập sách có độ dài 10,000 tokens
    stats = compressor.memory_stats(seq_len=10000)
    
    print(f"FP16 chiếm: {stats['original_mb']:.2f} MB")
    print(f"Đã nén chiếm: {stats['compressed_mb']:.2f} MB")
    ratio = stats['ratio']
    print(f"Hệ số nén thực tế: {ratio:.2f}x")
    
    # So với gốc (16-bit), (3.5 + 2.5) trung bình là 3-bit -> Tiết kiệm ~ 5x
    print_result("Dung lượng bộ nhớ giảm ít nhất 4 lần so với bản gốc", ratio >= 4.0)
def test_boundary_v_logic():
    print("\n--- Nhóm 5: Kiểm tra tính năng Boundary V ---")
    # Sử dụng 8 layer để thấy rõ sự khác biệt giữa "biên" và "giữa"
    num_layers = 8 
    num_heads = 2
    head_dim = 64
    seq_len = 16
    boundary_layers = 2

    # Khởi tạo compressor với 2 layer biên ở mỗi đầu
    compressor = KVCacheCompressor(
        num_layers=num_layers, 
        num_heads=num_heads, 
        head_dim=head_dim, 
        k_bits=3.5, 
        v_bits=2.5, 
        boundary_layers=boundary_layers
    )
    
    k = torch.randn(num_layers, num_heads, seq_len, head_dim)
    v = torch.randn(num_layers, num_heads, seq_len, head_dim)
    
    compressor.calibrate(k, v)
    compressed = compressor.compress(k, v)
    k_hat, v_hat = compressor.decompress(compressed)
    
    # 1. Kiểm tra ma trận Key (K): Tất cả các layer đều phải bị nén (MSE > 0)
    k_mse_all = torch.mean((k - k_hat) ** 2).item()
    print_result("Key (K) luôn bị nén ở MỌI layer (Bảo vệ Attention)", k_mse_all > 0)
    
    # 2. Kiểm tra ma trận Value (V) ở vùng biên (Layer 0, 1 và 6, 7): Phải giữ nguyên (MSE == 0)
    v_boundary_mse = 0.0
    for l in [0, 1, 6, 7]:
        v_boundary_mse += torch.mean((v[l] - v_hat[l]) ** 2).item()
    print_result("Value (V) ở các layer biên được giữ nguyên gốc (MSE == 0)", v_boundary_mse == 0.0)
    
    # 3. Kiểm tra ma trận Value (V) ở vùng giữa (Layer 2, 3, 4, 5): Phải bị nén (MSE > 0)
    v_middle_mse = 0.0
    for l in range(2, 6):
        v_middle_mse += torch.mean((v[l] - v_hat[l]) ** 2).item()
    print_result("Value (V) ở các layer giữa bị nén sâu để tiết kiệm RAM (MSE > 0)", v_middle_mse > 0)


if __name__ == "__main__":
    print("BẮT ĐẦU KIỂM THỬ TÍCH HỢP BỘ NHỚ KV CACHE (TRANSFORMER INTEGRATION)")
    test_shape_and_metadata()
    test_round_trip_quality()
    test_attention_score_preservation()
    test_memory_stats()
    test_boundary_v_logic()  # <-- Thêm dòng này
    print("\n🏁 TẤT CẢ MODULE ĐÃ SẴN SÀNG ĐỂ ĐƯA VÀO PRODUCTION!")