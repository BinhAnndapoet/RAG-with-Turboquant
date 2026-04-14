import torch
import math
from codebook import get_optimal_centroids, quantize_to_indices, dequantize_from_indices

def print_result(name, condition):
    if condition:
        print(f"✅ PASS: {name}")
    else:
        print(f"❌ FAIL: {name}")

def test_optimal_centroids():
    print("\n--- Nhóm 1: Kiểm tra các điểm neo Bài báo (Closed-form) ---")
    d_list = [64, 128, 256, 1024]
    
    # Tương đương 'test_1bit_centroids_match_paper'
    pass_1bit = True
    for d in d_list:
        centroids = get_optimal_centroids(1, d)
        expected = math.sqrt(2.0 / (math.pi * d))
        # Kiểm tra length = 2 và giá trị khớp công thức
        if len(centroids) != 2 or not math.isclose(centroids[1].item(), expected, rel_tol=1e-5):
            pass_1bit = False
    print_result("1-bit Centroids khớp chính xác công thức ±√(2/πd)", pass_1bit)

    # Tương đương 'test_2bit_centroids_match_paper'
    pass_2bit = True
    for d in d_list:
        centroids = get_optimal_centroids(2, d)
        expected = torch.tensor([-1.51, -0.453, 0.453, 1.51]) / math.sqrt(d)
        if len(centroids) != 4 or not torch.allclose(centroids, expected, rtol=1e-5):
            pass_2bit = False
    print_result("2-bit Centroids khớp bảng hằng số của TurboQuant", pass_2bit)

def test_lloyds_algorithm():
    print("\n--- Nhóm 2: Kiểm tra thuật toán Lloyd-Max (Động) ---")
    d = 64
    c_3bit = get_optimal_centroids(3, d)
    c_4bit = get_optimal_centroids(4, d)
    
    print_result("Sinh đúng 8 điểm neo cho cấu hình 3-bit", len(c_3bit) == 8)
    print_result("Sinh đúng 16 điểm neo cho cấu hình 4-bit", len(c_4bit) == 16)
    
    # Kiểm tra tính đối xứng của K-means trên đường cong Gauss
    sym_error_3 = abs(c_3bit[0].item() + c_3bit[-1].item())
    sym_error_4 = abs(c_4bit[0].item() + c_4bit[-1].item())
    print_result("Phân phối điểm neo đối xứng tuyệt đối (Sai số < 1e-4)", sym_error_3 < 1e-4 and sym_error_4 < 1e-4)

def test_quantization_indices():
    print("\n--- Nhóm 3: Kiểm tra phép ánh xạ ID (Quantization) ---")
    centroids = torch.tensor([-1.5, -0.5, 0.5, 1.5]) # Giả lập 2-bit
    
    # Tương đương 'test_batch_shape_preserved'
    values_batch = torch.randn(5, 10)
    indices_batch = quantize_to_indices(values_batch, centroids)
    print_result("Bảo toàn kích thước (Shape) khi xử lý Batch (5x10)", indices_batch.shape == (5, 10))

    # Tương đương 'test_all_indices_valid' (Out of bounds test)
    values_oob = torch.tensor([-99.0, 99.0])
    indices_oob = quantize_to_indices(values_oob, centroids)
    print_result("Xử lý an toàn giá trị vượt biên (Far left -> 0, Far right -> 3)", 
                 indices_oob[0].item() == 0 and indices_oob[1].item() == 3)

    # Tương đương 'test_matches_brute_force'
    # Thuật toán searchsorted siêu tốc của chúng ta phải khớp với cách tính trâu bò (tính khoảng cách từng điểm)
    torch.manual_seed(42)
    values_rand = torch.randn(500)
    
    # 1. Chạy hàm siêu tốc O(N log K) của chúng ta
    fast_indices = quantize_to_indices(values_rand, centroids)
    
    # 2. Chạy hàm Brute-force O(N * K)
    # Lấy từng giá trị trừ đi từng centroid, lấy abs, rồi tìm index nhỏ nhất
    brute_indices = torch.argmin(torch.abs(values_rand.unsqueeze(1) - centroids.unsqueeze(0)), dim=1)
    
    match = torch.equal(fast_indices, brute_indices)
    print_result("Hàm Searchsorted (Tốc độ cao) khớp kết quả 100% với hàm Brute-force", match)

def test_dequantization():
    print("\n--- Nhóm 4: Kiểm tra phép khôi phục (Dequantization) ---")
    centroids = torch.tensor([-1.5, -0.5, 0.5, 1.5])
    original_indices = torch.tensor([0, 2, 3, 1, 0])
    
    # Giải mã từ ID về số thập phân
    restored_values = dequantize_from_indices(original_indices, centroids)
    expected_values = torch.tensor([-1.5, 0.5, 1.5, -0.5, -1.5])
    
    match = torch.equal(restored_values, expected_values)
    print_result("Giải nén (Dequantize) khôi phục chuẩn xác mảng Tensor thực", match)


if __name__ == "__main__":
    print("BẮT ĐẦU KIỂM THỬ MODULE LƯỢNG TỬ HÓA (QUANTIZER BUILDER)")
    test_optimal_centroids()
    test_lloyds_algorithm()
    test_quantization_indices()
    test_dequantization()
    print("\n✅ HOÀN TẤT KIỂM THỬ.")