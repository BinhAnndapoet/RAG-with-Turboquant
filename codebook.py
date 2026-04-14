import torch
import math

def _stable_gaussian_cdf(x: torch.Tensor) -> torch.Tensor:
    """
    Tính hàm phân phối tích lũy (CDF) của phân phối chuẩn tắc N(0, 1) một cách ổn định.

    Thay vì sử dụng hàm CDF thông thường dễ bị thiếu chính xác số học (underflow) 
    ở các vùng đuôi (tail regions), hàm này sử dụng hàm sai số bù (complementary 
    error function - erfc) để duy trì độ chính xác cực cao khi `x` có giá trị tuyệt đối lớn.

    Tham số:
        x (torch.Tensor): Tensor chứa các giá trị đầu vào.

    Trả về:
        torch.Tensor: Tensor chứa các giá trị xác suất tích lũy tương ứng, cùng shape với `x`.
    """
    # Công thức: Phi(x) = 0.5 * erfc(-x / sqrt(2))
    # Phép chia cho sqrt(2) là để chuẩn hóa phương sai về hàm erfc tiêu chuẩn.
    return 0.5 * torch.special.erfc(-x / math.sqrt(2.0))


def _lloyds_gaussian_vectorized(
    n_centroids: int, 
    sigma: float, 
    n_iter: int = 100, 
    device: str | torch.device = 'cpu'
) -> torch.Tensor:
    """
    Chạy thuật toán Lloyd-Max dạng vector hóa để tìm các điểm đại diện (centroids) tối ưu.

    Thuật toán lặp lại quá trình tính toán ranh giới (boundaries) chia cắt dữ liệu 
    và cập nhật lại các điểm đại diện sao cho giảm thiểu sai số lượng tử hóa (MSE) 
    đối với phân phối chuẩn Gaussian N(0, sigma^2).

    Tham số:
        n_centroids (int): Số lượng điểm đại diện (centroids) cần tìm.
        sigma (float): Độ lệch chuẩn của phân phối Gaussian mục tiêu.
        n_iter (int, tùy chọn): Số vòng lặp tối đa của thuật toán. Mặc định là 100.

    Trả về:
        torch.Tensor: Tensor chứa các giá trị centroid tối ưu, kiểu float32.
    """
    n = n_centroids
    
    # Khởi tạo các ranh giới ban đầu dựa trên phân vị (quantiles) của phân phối chuẩn.
    # Sử dụng float64 để giảm thiểu sai số tích lũy trong quá trình lặp.
    p = torch.linspace(0, 1, n + 1, dtype=torch.float64, device=device)[1:-1]
    dist = torch.distributions.Normal(0.0, 1.0)
    boundaries = dist.icdf(p) * sigma

    for _ in range(n_iter):
        # 1. Ghép thêm ranh giới vô cực (-inf, +inf) cho 2 vùng ngoài cùng (buckets)
        a = torch.cat([torch.tensor([-float('inf')], dtype=torch.float64, device=device), boundaries])
        b = torch.cat([boundaries, torch.tensor([float('inf')], dtype=torch.float64, device=device)])

        # Chuẩn hóa ranh giới về phân phối N(0, 1) để tính PDF/CDF
        a_std = a / sigma
        b_std = b / sigma

        # 2. Tính PDF tại các ranh giới: phi(x) = exp(-x^2/2) / sqrt(2*pi)
        pdf_a = torch.exp(-0.5 * a_std**2) / math.sqrt(2.0 * math.pi)
        pdf_b = torch.exp(-0.5 * b_std**2) / math.sqrt(2.0 * math.pi)

        # 3. Tính khối lượng xác suất (Probability mass) của từng bucket
        # Sử dụng CDF ổn định để tránh prob = 0 ở các vùng đuôi xa
        prob = _stable_gaussian_cdf(b_std) - _stable_gaussian_cdf(a_std)
        
        centroids = torch.zeros_like(a)
        mask = prob > 1e-18  # Lọc các bucket có xác suất hợp lệ để tránh chia cho 0
        
        # 4. Cập nhật Centroids theo kỳ vọng có điều kiện: E[X | a < X < b]
        # Công thức giải tích cho Gaussian: mu + sigma * (pdf(a) - pdf(b)) / (CDF(b) - CDF(a))
        centroids[mask] = sigma * (pdf_a[mask] - pdf_b[mask]) / prob[mask]

        # Xử lý các bucket ngoại biên (outliers) có xác suất xấp xỉ 0
        inf_a = torch.isinf(a)
        inf_b = torch.isinf(b)
        centroids[~mask & inf_a] = b[~mask & inf_a] - sigma
        centroids[~mask & inf_b] = a[~mask & inf_b] + sigma

        # 5. Cập nhật lại ranh giới chia (midpoints) nằm chính giữa 2 centroids kề nhau
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

    return centroids.to(torch.float32)


def get_optimal_centroids(bit_width: int, d: int, device: str | torch.device = 'cpu') -> torch.Tensor:
    """
    Sinh từ điển mã (Codebook) tối ưu dựa trên số bit lượng tử hóa.

    Đối với các trường hợp 1-bit và 2-bit, hàm sử dụng các giá trị tĩnh (hardcoded) 
    đã được chứng minh là tối ưu trên lý thuyết nhằm tiết kiệm chi phí tính toán. 
    Với số bit lớn hơn, hàm gọi thuật toán Lloyd-Max để tìm nghiệm xấp xỉ.

    Tham số:
        bit_width (int): Số bit dùng để biểu diễn mỗi phần tử (ví dụ: 1, 2, 4, 8).
        d (int): Chiều dữ liệu, được dùng để tính toán độ lệch chuẩn (sigma = 1/sqrt(d)).
        device (str | torch.device, tùy chọn): Thiết bị xử lý. Mặc định là 'cpu'.

    Trả về:
        torch.Tensor: Tensor chứa các giá trị codebook (centroids).
    """
    n_centroids = 1 << bit_width  # Tương đương 2^bit_width
    
    # Trường hợp đặc biệt 1-bit: Nghiệm giải tích là +- sqrt(2/(pi*d))
    if bit_width == 1:
        c = math.sqrt(2.0 / (math.pi * d))
        return torch.tensor([-c, c], dtype=torch.float32, device=device)
        
    # Trường hợp đặc biệt 2-bit: Sử dụng hằng số chuẩn hóa tối ưu đã biết
    if bit_width == 2:
        return torch.tensor([-1.51, -0.453, 0.453, 1.51], dtype=torch.float32, device=device) / math.sqrt(d)

    # Các trường hợp > 2 bit: Khởi chạy thuật toán tìm kiếm Lloyd-Max
    sigma = 1.0 / math.sqrt(d)
    return _lloyds_gaussian_vectorized(n_centroids, sigma=sigma, device=device)


def quantize_to_indices(values: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """
    Lượng tử hóa các giá trị thực thành chỉ số (indices) của điểm đại diện gần nhất.

    Sử dụng tìm kiếm nhị phân (binary search) thông qua `torch.searchsorted` 
    lên các điểm ranh giới (midpoints) giữa các centroids để đạt tốc độ cao.

    Tham số:
        values (torch.Tensor): Tensor chứa các giá trị thực cần lượng tử hóa.
        centroids (torch.Tensor): Mảng các điểm đại diện (đã được sắp xếp tăng dần).

    Trả về:
        torch.Tensor: Tensor kiểu số nguyên chứa ID (index) của centroid tương ứng.
    """
    # Tính ranh giới chia cắt (midpoints) giữa các cụm
    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    
    # Ánh xạ giá trị vào các khoảng (buckets) để lấy chỉ số
    return torch.searchsorted(boundaries, values.contiguous())


def dequantize_from_indices(indices: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """
    Giải lượng tử hóa (Dequantize) từ chỉ số nguyên về lại giá trị thực.

    Tham số:
        indices (torch.Tensor): Tensor chứa các chỉ số (ID) đã được lượng tử hóa.
        centroids (torch.Tensor): Từ điển mã (Codebook) chứa các giá trị thực tương ứng.

    Trả về:
        torch.Tensor: Tensor chứa các giá trị thực (float32) sau khi giải nén, 
            có cùng shape với `indices`.
    """
    # Thực hiện Lookup (ánh xạ trực tiếp) từ bộ mã Codebook
    return centroids[indices]