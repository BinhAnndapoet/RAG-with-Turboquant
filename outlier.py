import torch
import math
from dataclasses import dataclass
from polar_quant import PolarQuant

@dataclass
class OutlierCompressedKV:
    """
    Lớp dữ liệu (Dataclass) lưu trữ trạng thái nén của bộ đệm KV (KV Cache).
    
    Cấu trúc này chia dữ liệu thành hai nhánh dựa trên độ quan trọng của kênh đặc trưng:
    nhánh ngoại lai (outlier - sử dụng nhiều bit hơn) và nhánh thông thường (normal).

    Thuộc tính:
        outlier_indices (torch.Tensor): Chỉ số lượng tử hóa của các kênh ngoại lai.
        outlier_norms (torch.Tensor): Chuẩn L2 nguyên bản của các kênh ngoại lai.
        normal_indices (torch.Tensor): Chỉ số lượng tử hóa của các kênh thông thường.
        normal_norms (torch.Tensor): Chuẩn L2 nguyên bản của các kênh thông thường.
    """
    outlier_indices: torch.Tensor
    outlier_norms: torch.Tensor
    normal_indices: torch.Tensor
    normal_norms: torch.Tensor

def _compute_channel_split(d: int, target_bits: float) -> tuple[int, int, int, int]:
    """
    Tính toán chiến lược phân bổ ngân sách bit (Bit Allocation) cho các kênh.

    Thuật toán phân bổ số lượng kênh ngoại lai (outliers) và kênh thông thường (normals) 
    sao cho trung bình (expected bit-rate) đạt đúng mục tiêu `target_bits`.

    Ví dụ: d=128, target_bits=2.5. 
    Tổng ngân sách = 320 bits. Mức bit thấp (low) = 2, mức bit cao (high) = 3.
    Cần phân bổ 64 kênh x 3-bit (192) + 64 kênh x 2-bit (128) = 320 bits.

    Tham số:
        d (int): Tổng số chiều (kênh) của dữ liệu.
        target_bits (float): Số bit trung bình mục tiêu cho mỗi phần tử.

    Trả về:
        tuple[int, int, int, int]: Gồm 4 giá trị:
            - n_outlier: Số lượng kênh ngoại lai.
            - high_bits: Số bit cấp phát cho kênh ngoại lai.
            - n_normal: Số lượng kênh thông thường.
            - low_bits: Số bit cấp phát cho kênh thông thường.
    """
    total_bits = round(d * target_bits)
    low_bits = math.floor(target_bits)
    high_bits = low_bits + 1
    
    # Số lượng kênh ngoại lai được tính bằng phần dư của tổng ngân sách bit
    n_outlier = total_bits - (d * low_bits)
    n_normal = d - n_outlier
    
    return n_outlier, high_bits, n_normal, low_bits


class OutlierAwareQuantizer:
    """
    Bộ lượng tử hóa nhận biết ngoại lai (Outlier-Aware Quantizer).

    Đặc trưng của các Mô hình Ngôn ngữ Lớn (LLMs) là sự xuất hiện của các giá trị ngoại lai 
    rất lớn ở một vài kênh (channels) cố định. Module này tự động xác định các kênh quan trọng 
    này và áp dụng độ phân giải lượng tử hóa cao hơn (high-bit) để bảo toàn thông tin, 
    trong khi nén mạnh hơn (low-bit) ở các kênh còn lại.

    Thuộc tính:
        d (int): Chiều dữ liệu đầu vào.
        target_bits (float): Tốc độ bit trung bình mục tiêu.
    """
    def __init__(self, d: int, target_bits: float, device: str | torch.device = 'cpu', seed: int = 42):
        """Khởi tạo bộ lượng tử hóa phân tầng dựa trên ngân sách bit."""
        self.d = d
        self.target_bits = target_bits
        self.device = device
        
        # 1. Tính toán cấu trúc phân chia kênh đặc trưng
        self.n_outlier, self.high_bits, self.n_normal, self.low_bits = _compute_channel_split(d, target_bits)
        
        # 2. Khởi tạo 2 module PolarQuant độc lập cho 2 mức bit giải mã
        self.pq_outlier = PolarQuant(
            d=self.n_outlier, bit_width=self.high_bits, device=device, seed=seed
        ) if self.n_outlier > 0 else None
        
        self.pq_normal = PolarQuant(
            d=self.n_normal, bit_width=self.low_bits, device=device, seed=seed
        ) if self.n_normal > 0 else None
        
        # Bộ nhớ lưu vị trí (indices) của các kênh ngoại lai và bình thường sau quá trình Calibration
        self.outlier_idx = None
        self.normal_idx = None

    def fit(self, x: torch.Tensor) -> None:
        """
        Hiệu chỉnh (Calibrate) bộ lượng tử hóa dựa trên dữ liệu mẫu.

        Trích xuất vị trí của các kênh ngoại lai bằng cách đo lường biên độ (magnitude) 
        tuyệt đối lớn nhất dọc theo các kênh (channel dimension) của dữ liệu mẫu.

        Tham số:
            x (torch.Tensor): Tensor dữ liệu hiệu chỉnh (Calibration dataset).
        """
        # Bước 1: Tìm biên độ cực đại dọc theo trục chuỗi (sequence/batch dimension)
        if x.dim() > 1:
            channel_magnitudes, _ = x.abs().max(dim=0)
        else:
            channel_magnitudes = x.abs()
            
        # Bước 2: Chọn top-K kênh có biên độ lớn nhất làm kênh ngoại lai
        if self.n_outlier > 0:
            _, self.outlier_idx = torch.topk(channel_magnitudes, self.n_outlier)
            # Sắp xếp lại index để duy trì trật tự không gian ban đầu của tensor
            self.outlier_idx = torch.sort(self.outlier_idx)[0] 
            
            # Khởi tạo mặt nạ (mask) logic để suy luận ra các kênh thông thường còn lại
            all_idx = torch.arange(self.d, device=self.device)
            mask = torch.ones(self.d, dtype=torch.bool, device=self.device)
            mask[self.outlier_idx] = False
            self.normal_idx = all_idx[mask]
        else:
            self.normal_idx = torch.arange(self.d, device=self.device)

    def quantize(self, x: torch.Tensor) -> OutlierCompressedKV:
        """
        Thực hiện phân tách không gian đặc trưng và lượng tử hóa dữ liệu.

        Tham số:
            x (torch.Tensor): Tensor đầu vào cần lượng tử hóa.

        Trả về:
            OutlierCompressedKV: Đối tượng chứa dữ liệu đã nén phân nhánh.

        Ngoại lệ:
            RuntimeError: Ném ra nếu gọi hàm này khi chưa chạy `fit()` trước đó.
        """
        if self.outlier_idx is None:
            raise RuntimeError("Cần gọi phương thức fit() bằng dữ liệu mẫu trước khi thực hiện quantize().")
            
        outlier_indices, outlier_norms, normal_indices, normal_norms = None, None, None, None
        
        # Phân tách và nén nhánh ngoại lai (Outliers) với bit-width cao
        if self.n_outlier > 0:
            x_out = x[..., self.outlier_idx]
            outlier_indices, outlier_norms = self.pq_outlier.quantize(x_out)
            
        # Phân tách và nén nhánh thông thường (Normals) với bit-width thấp
        if self.n_normal > 0:
            x_norm = x[..., self.normal_idx]
            normal_indices, normal_norms = self.pq_normal.quantize(x_norm)
            
        return OutlierCompressedKV(outlier_indices, outlier_norms, normal_indices, normal_norms)

    def dequantize(self, compressed: OutlierCompressedKV) -> torch.Tensor:
        """
        Giải lượng tử hóa và tái tổ hợp (reconstruct) không gian đặc trưng gốc.

        Tham số:
            compressed (OutlierCompressedKV): Dữ liệu phân nhánh đã được lượng tử hóa.

        Trả về:
            torch.Tensor: Tensor xấp xỉ đã được giải nén với kích thước nguyên bản.
        """
        # Khởi tạo một tensor đệm với các giá trị 0, giữ nguyên cấu trúc batch ban đầu
        batch_shape = compressed.normal_indices.shape[:-1] if compressed.normal_indices is not None else compressed.outlier_indices.shape[:-1]
        x_hat = torch.zeros((*batch_shape, self.d), dtype=torch.float32, device=self.device)
        
        # Giải nén và ánh xạ nhánh ngoại lai về đúng index ban đầu
        if self.n_outlier > 0:
            x_out_hat = self.pq_outlier.dequantize(compressed.outlier_indices, compressed.outlier_norms)
            x_hat[..., self.outlier_idx] = x_out_hat
            
        # Giải nén và ánh xạ nhánh thông thường về các index còn lại
        if self.n_normal > 0:
            x_norm_hat = self.pq_normal.dequantize(compressed.normal_indices, compressed.normal_norms)
            x_hat[..., self.normal_idx] = x_norm_hat
            
        return x_hat