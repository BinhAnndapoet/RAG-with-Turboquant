import torch
from polar_quant import PolarQuant

def print_result(name, condition):
    if condition:
        print(f"✅ PASS: {name}")
    else:
        print(f"❌ FAIL: {name}")

def test_round_trip_mse_bounds():
    print("\n--- Nhóm 1: Kiểm tra sai số nén so với ngưỡng của bài báo ---")
    # Các ngưỡng MSE tối đa cho phép từ Table 2 của bài báo gốc
    # 1-bit: 0.36, 2-bit: 0.117, 3-bit: 0.03
    expected_mse_bounds = {1: 0.36, 2: 0.117, 3: 0.03}
    
    dimensions = [64, 128, 256]
    bit_widths = [1, 2, 3]
    n_samples = 200 # Số mẫu thử để tính trung bình
    
    torch.manual_seed(42)
    
    for b in bit_widths:
        for d in dimensions:
            pq = PolarQuant(d=d, bit_width=b)
            total_mse = 0.0
            
            for _ in range(n_samples):
                # Tạo vector đơn vị ngẫu nhiên (Unit-norm vectors)
                x = torch.randn(1, d)
                x = x / torch.norm(x, dim=-1, keepdim=True)
                
                indices, norms = pq.quantize(x)
                x_hat = pq.dequantize(indices, norms)
                
                total_mse += torch.mean((x - x_hat)**2).item()
            
            avg_mse = total_mse / n_samples
            # Cho phép sai số gấp 2 lần ngưỡng lý thuyết do tính ngẫu nhiên của Rotation
            threshold = expected_mse_bounds[b] * 2.0
            
            status = avg_mse < threshold
            print(f"[{b}-bit, d={d}] Avg MSE: {avg_mse:.4f} (Ngưỡng: {threshold:.4f})")
            print_result(f"Sai số nằm trong giới hạn cho phép", status)

def test_indices_range():
    print("\n--- Nhóm 2: Kiểm tra tính hợp lệ của chỉ số (Indices) ---")
    # Đảm bảo các ID nén nằm đúng trong khoảng [0, 2^bit - 1]
    d = 128
    bit_width = 3
    pq = PolarQuant(d=d, bit_width=bit_width)
    
    x = torch.randn(10, d)
    indices, _ = pq.quantize(x)
    
    min_idx = indices.min().item()
    max_idx = indices.max().item()
    expected_max = (1 << bit_width) - 1
    
    print_result(f"Chỉ số nhỏ nhất >= 0 ({min_idx})", min_idx >= 0)
    print_result(f"Chỉ số lớn nhất <= {expected_max} ({max_idx})", max_idx <= expected_max)

def test_norm_correction_effectiveness():
    print("\n--- Nhóm 3: Kiểm tra hiệu quả của Norm Correction ---")
    # Theo file gốc, Norm Correction giúp đưa vector về gần Norm 1 hơn
    d = 256
    torch.manual_seed(99)
    x = torch.randn(1, d)
    x = x / torch.norm(x)
    
    # 1. Không dùng hiệu chỉnh
    pq_no_corr = PolarQuant(d=d, bit_width=2, norm_correction=False)
    idx_nc, norm_nc = pq_no_corr.quantize(x)
    x_hat_nc = pq_no_corr.dequantize(idx_nc, norm_nc)
    err_no_corr = abs(torch.norm(x_hat_nc).item() - 1.0)
    
    # 2. Có dùng hiệu chỉnh
    pq_corr = PolarQuant(d=d, bit_width=2, norm_correction=True)
    idx_c, norm_c = pq_corr.quantize(x)
    x_hat_c = pq_corr.dequantize(idx_c, norm_c)
    err_corr = abs(torch.norm(x_hat_c).item() - 1.0)
    
    print(f"Sai số Norm (Không hiệu chỉnh): {err_no_corr:.6f}")
    print(f"Sai số Norm (Có hiệu chỉnh): {err_corr:.6f}")
    
    print_result("Norm Correction giúp bảo toàn độ dài vector tốt hơn", err_corr < err_no_corr)

if __name__ == "__main__":
    print("BẮT ĐẦU KIỂM THỬ HỆ THỐNG POLAR QUANT (PYTORCH ENGINE)")
    test_round_trip_mse_bounds()
    test_indices_range()
    test_norm_correction_effectiveness()
    print("\n🏁 TẤT CẢ CÁC BÀI KIỂM TRA ĐÃ HOÀN TẤT.")