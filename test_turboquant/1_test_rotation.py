import torch
import math
from rotation import fast_walsh_hadamard_transform, get_random_signs, rotate_tensor, inverse_rotate_tensor

def print_result(name: str, condition: bool) -> None:
    """
    In kết quả của một ca kiểm thử (test case) ra màn hình standard output.

    Tham số:
        name (str): Tên hoặc mô tả kỹ thuật của chức năng đang được kiểm thử.
        condition (bool): Biểu thức logic đánh giá kết quả (True tương ứng với PASS, False là FAIL).
    """
    if condition:
        print(f"✅ PASS: {name}")
    else:
        print(f"❌ FAIL: {name}")

def test_lossless_reconstruction() -> None:
    """
    Kiểm thử tính toàn vẹn dữ liệu (Lossless Reconstruction) và tính tự nghịch đảo.

    Phép biến đổi Walsh-Hadamard (đã chuẩn hóa) có tính chất tự nghịch đảo (Involutory), 
    nghĩa là áp dụng hai lần liên tiếp sẽ hoàn trả lại ma trận định danh (H * H = I).
    Hàm này xác minh rằng dữ liệu qua quá trình `rotate` và `inverse_rotate` không 
    bị thất thoát thông tin do sai số dấu phẩy động (floating-point error).
    """
    print("\n--- Nhóm 1: Tính toàn vẹn của thuật toán ---")
    torch.manual_seed(42)
    d = 128
    
    # Giả lập một mini-batch của bộ nhớ KV Cache: [Batch=2, Heads=4, Head_Dim=128]
    x_original = torch.randn(2, 4, d) 
    
    # Kiểm chứng 1: Tính tự nghịch đảo của Fast Walsh-Hadamard Transform (FWHT)
    x_fwht2 = fast_walsh_hadamard_transform(fast_walsh_hadamard_transform(x_original))
    mse_fwht = torch.mean((x_original - x_fwht2) ** 2).item()
    print_result("Tính chất tự nghịch đảo của WHT (Involutory)", mse_fwht < 1e-10)

    # Kiểm chứng 2: Toàn vẹn hệ thống xoay hoàn chỉnh (Lật dấu + FWHT)
    x_rotated = rotate_tensor(x_original)
    x_restored = inverse_rotate_tensor(x_rotated)
    mse_full = torch.mean((x_original - x_restored) ** 2).item()
    print_result("Khôi phục toàn vẹn 100% sau khi xoay", mse_full < 1e-10)

def test_mathematical_properties() -> None:
    """
    Kiểm thử các bất biến toán học (Mathematical Invariants) của phép xoay trực giao.

    Để đảm bảo mô hình LLM vẫn tính toán đúng điểm số Attention sau khi dữ liệu 
    bị xoay, phép biến đổi bắt buộc phải là một phép đẳng cự (Isometry), tức là 
    phải bảo toàn khoảng cách/độ dài (L2 Norm) và góc giữa các vector (Inner Product).
    """
    print("\n--- Nhóm 2: Các tính chất Toán học cốt lõi ---")
    d = 64
    x = torch.randn(d)
    y = torch.randn(d)
    signs = get_random_signs(d, seed=99)
    
    x_rot = rotate_tensor(x, signs)
    y_rot = rotate_tensor(y, signs)

    # 1. Bảo toàn năng lượng (Preserves L2 Norm)
    # Tổng năng lượng của vector không được phép khuếch đại hoặc suy giảm.
    norm_x = torch.norm(x).item()
    norm_x_rot = torch.norm(x_rot).item()
    print_result(f"Giữ nguyên độ dài Vector (Norm): {norm_x:.4f} == {norm_x_rot:.4f}", 
                 math.isclose(norm_x, norm_x_rot, rel_tol=1e-5))

    # 2. Bảo toàn Tích vô hướng (Preserves Inner Product)
    # Cơ chế Attention (Q * K^T) hoạt động dựa trên tích vô hướng. Nếu phép xoay 
    # làm thay đổi tích vô hướng, mô hình sẽ hoàn toàn mất khả năng suy luận ngữ cảnh.
    ip_original = torch.dot(x, y).item()
    ip_rotated = torch.dot(x_rot, y_rot).item()
    print_result(f"Giữ nguyên Tích vô hướng (Inner Product): {ip_original:.4f} == {ip_rotated:.4f}", 
                 math.isclose(ip_original, ip_rotated, rel_tol=1e-5))

def test_batch_vs_single() -> None:
    """
    Kiểm chứng độ chính xác của cơ chế lan truyền (Broadcasting) và xử lý vector hóa.

    Phép toán vector hóa trên toàn bộ Batch (Batch Processing) thông qua GPU phải cho 
    ra kết quả chuẩn xác tới từng bit so với việc tính toán tuần tự trên từng vector 
    (Single Processing) nhằm đảm bảo hệ thống có thể mở rộng (scale) tính toán song song.
    """
    print("\n--- Nhóm 3: Khả năng xử lý song song (Broadcasting) ---")
    batch_size = 10
    d = 64
    X_batch = torch.randn(batch_size, d)
    signs = get_random_signs(d, seed=42)

    # Xử lý Vector hóa: Thực thi một lần trên toàn bộ ma trận (Tối ưu cho GPU)
    batch_rotated = rotate_tensor(X_batch, signs)

    # Xử lý Tuần tự: Chạy qua vòng lặp for trên từng vector (Dùng làm baseline đối chiếu)
    single_rotated_list = []
    for i in range(batch_size):
        single_rot = rotate_tensor(X_batch[i], signs)
        single_rotated_list.append(single_rot)
    
    single_rotated_stacked = torch.stack(single_rotated_list)

    # Đối chiếu chéo để đảm bảo broadcasting không làm thay đổi ngữ nghĩa tính toán
    mse_batch = torch.mean((batch_rotated - single_rotated_stacked) ** 2).item()
    print_result("Xử lý Batch (Đa luồng) khớp 100% với xử lý Single", mse_batch < 1e-10)

def test_edge_cases() -> None:
    """
    Kiểm thử khả năng kiểm soát ngoại lệ (Exception Handling) đối với các trường hợp biên.
    
    Hàm FWHT dạng cánh bướm (Butterfly) được cài đặt với yêu cầu nghiêm ngặt về 
    kích thước số chiều (dimension): bắt buộc phải là một lũy thừa của 2. Ca kiểm thử 
    này đánh giá xem hệ thống có chủ động bắt lỗi khi vi phạm điều kiện này hay không.
    """
    print("\n--- Nhóm 4: Bắt lỗi ngoại lệ (Edge Cases) ---")
    try:
        # Truyền vào kích thước d=100 (Không phải là lũy thừa của 2)
        fast_walsh_hadamard_transform(torch.randn(3, 100)) 
        print_result("Chặn vector không phải lũy thừa của 2", False)
    except AssertionError:
        # Kỳ vọng thuật toán sẽ ném ra lỗi AssertionError để chặn luồng thực thi
        print_result("Chặn vector không phải lũy thừa của 2", True)

def test_outlier_smashing() -> None:
    """
    Kiểm thử đặc tính cốt lõi của không gian xoay: Phân tán trị ngoại lai (Outlier Dispersion).
    
    Trong mô hình ngôn ngữ lớn (LLMs), các đặc trưng (features) thường chứa các trị 
    ngoại lai khổng lồ làm hỏng các bộ lượng tử hóa thông thường. Phép xoay sẽ phân tán 
    năng lượng của một cực trị (outlier) duy nhất và san đều (smash) nó ra toàn bộ 
    các chiều dữ liệu khác, giúp làm giảm triệt để giá trị tuyệt đối lớn nhất (Max Absolute Value).
    """
    print("\n--- Nhóm 5: Tính năng nghiền nát ngoại lai (Đặc thù TurboQuant) ---")
    x = torch.ones(64) * 0.1
    # Bơm vào hệ thống một trị ngoại lai khổng lồ (Needle)
    x[15] = 100.0  
    
    # Áp dụng ma trận xoay để phân tán năng lượng
    x_rotated = rotate_tensor(x)
    
    # Kiểm tra xem biên độ cực đại đã bị giảm xuống mức an toàn cho lượng tử hóa hay chưa
    max_val = x_rotated.abs().max().item()
    print_result(f"San phẳng Outlier 100.0 xuống còn Max = {max_val:.2f}", max_val < 20.0)

if __name__ == "__main__":
    print("BẮT ĐẦU KIỂM THỬ MODULE WHT/ROTATION (PYTORCH ENGINE)")
    test_lossless_reconstruction()
    test_mathematical_properties()
    test_batch_vs_single()
    test_edge_cases()
    test_outlier_smashing()
    print("\n HOÀN TẤT!")