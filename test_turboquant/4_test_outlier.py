import torch
import math
from outlier import OutlierAwareQuantizer, _compute_channel_split

def print_result(name, condition):
    if condition:
        print(f"✅ PASS: {name}")
    else:
        print(f"❌ FAIL: {name}")

def test_effective_bit_rate():
    print("\n--- Nhóm 1: Kiểm tra Tỷ lệ Bit Hiệu dụng (Fractional Bits) ---")
    d = 128
    
    # 1. Test 2.5-bit
    n_out_25, h_bits_25, n_norm_25, l_bits_25 = _compute_channel_split(d, 2.5)
    effective_bits_25 = (n_out_25 * h_bits_25 + n_norm_25 * l_bits_25) / d
    print_result(f"Cấu hình 2.5-bit chia kênh chuẩn xác (Thực tế: {effective_bits_25})", 
                 math.isclose(effective_bits_25, 2.5, rel_tol=1e-5))

    # 2. Test 3.5-bit
    n_out_35, h_bits_35, n_norm_35, l_bits_35 = _compute_channel_split(d, 3.5)
    effective_bits_35 = (n_out_35 * h_bits_35 + n_norm_35 * l_bits_35) / d
    print_result(f"Cấu hình 3.5-bit chia kênh chuẩn xác (Thực tế: {effective_bits_35})", 
                 math.isclose(effective_bits_35, 3.5, rel_tol=1e-5))

def test_round_trip_quality():
    print("\n--- Nhóm 2: Kiểm tra Sai số toàn trình (Round Trip Quality) ---")
    # Đảm bảo MSE bị chặn (bounded MSE) khi nén bằng thuật toán chia kênh
    d = 128
    torch.manual_seed(42)
    oq = OutlierAwareQuantizer(d=d, target_bits=3.5, seed=42)
    
    x = torch.randn(1, d)
    oq.fit(x) # Phải gọi fit trước khi quantize
    compressed = oq.quantize(x)
    x_hat = oq.dequantize(compressed)
    
    mse = torch.mean((x - x_hat)**2).item()
    print(f"Sai số MSE cho 3.5-bit: {mse:.6f}")
    print_result("Sai số MSE nằm trong giới hạn an toàn (< 0.05)", mse < 0.05)

def test_outliers_selected():
    print("\n--- Nhóm 3: Kiểm tra Khả năng Chọn đúng Ngoại lai (Outliers Selected) ---")
    # Các kênh có giá trị lớn nhất phải được ưu tiên chọn làm Outlier
    d = 128
    torch.manual_seed(99)
    x = torch.randn(1, d) * 0.1
    
    # Bơm 2 giá trị khổng lồ vào vị trí 10 và 100
    x[0, 10] = 50.0
    x[0, 100] = -45.0
    
    oq = OutlierAwareQuantizer(d=d, target_bits=2.5, seed=42)
    oq.fit(x)
    
    # Kiểm tra xem index 10 và 100 có nằm trong danh sách outlier_idx không
    outlier_list = oq.outlier_idx.tolist()
    print_result("Hệ thống tự động tóm gọn chính xác các kênh Outlier khổng lồ", 
                 10 in outlier_list and 100 in outlier_list)

def test_batch_matches_single():
    print("\n--- Nhóm 4: Kiểm tra Xử lý Đa luồng (Batch vs Single) ---")
    # Đảm bảo nén cả cụm (Batch) cho ra kết quả y hệt nén từng vector
    d = 128
    batch_size = 5
    torch.manual_seed(7)
    X = torch.randn(batch_size, d)
    
    oq = OutlierAwareQuantizer(d=d, target_bits=3.5, seed=42)
    oq.fit(X) # Quét Outlier trên toàn bộ Batch
    
    # Nén theo Batch
    batch_compressed = oq.quantize(X)
    batch_recon = oq.dequantize(batch_compressed)
    
    # Nén từng dòng
    single_recon_list = []
    for i in range(batch_size):
        single_compressed = oq.quantize(X[i:i+1])
        single_recon = oq.dequantize(single_compressed)
        single_recon_list.append(single_recon)
        
    single_recon_stacked = torch.cat(single_recon_list, dim=0)
    
    match = torch.allclose(batch_recon, single_recon_stacked, atol=1e-6)
    print_result("Chế độ xử lý Batch chạy chuẩn xác 100% so với chạy Single", match)

def test_deterministic():
    print("\n--- Nhóm 5: Kiểm tra Tính Tất định (Deterministic) ---")
    # Cùng một seed phải cho ra kết quả xoay và nén y hệt nhau
    d = 128
    torch.manual_seed(1)
    x = torch.randn(1, d)
    
    oq1 = OutlierAwareQuantizer(d=d, target_bits=3.5, seed=42)
    oq2 = OutlierAwareQuantizer(d=d, target_bits=3.5, seed=42)
    
    oq1.fit(x)
    oq2.fit(x)
    
    x_hat_1 = oq1.dequantize(oq1.quantize(x))
    x_hat_2 = oq2.dequantize(oq2.quantize(x))
    
    match = torch.equal(x_hat_1, x_hat_2)
    print_result("Khởi tạo cùng Seed cho ra cấu trúc nén giống nhau tuyệt đối", match)


if __name__ == "__main__":
    print("BẮT ĐẦU KIỂM THỬ MODULE TỐI ƯU HÓA NGOẠI LAI (OUTLIER)")
    test_effective_bit_rate()
    test_round_trip_quality()
    test_outliers_selected()
    test_batch_matches_single()
    test_deterministic()
    print("\n🏁 HOÀN TẤT TẤT CẢ KIỂM THỬ CORE!")