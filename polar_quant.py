import torch
from rotation import rotate_tensor, inverse_rotate_tensor, get_random_signs
from codebook import get_optimal_centroids, quantize_to_indices, dequantize_from_indices

class PolarQuant:
    """
    Hệ thống lượng tử hóa vector theo Tọa độ cực (Polar Coordinate Quantization).

    Hệ thống này tách biệt biên độ (L2 norm) và hướng của vector. Bằng cách áp dụng 
    phép xoay ngẫu nhiên (Random Rotation) lên các vector đã được chuẩn hóa, hệ thống 
    buộc dữ liệu phải tuân theo phân phối tiệm cận chuẩn (Gaussian-like), từ đó tối đa hóa 
    hiệu năng của bộ lượng tử hóa vô hướng (Scalar Quantization). Phương pháp này đặc biệt 
    hiệu quả để nén bộ đệm KV (KV Cache) trong các mô hình RAG có ngữ cảnh dài.

    Thuộc tính:
        d (int): Kích thước của chiều dữ liệu (feature dimension).
        bit_width (int): Số lượng bit dùng để mã hóa mỗi phần tử.
        norm_correction (bool): Cờ kiểm soát việc áp dụng hiệu chỉnh chuẩn L2 trong quá trình giải mã.
        signs (torch.Tensor): Vector dấu ngẫu nhiên dùng cho phép biến đổi Hadamard.
        centroids (torch.Tensor): Từ điển mã (Codebook) chứa các điểm đại diện tối ưu.
    """

    def __init__(
        self, 
        d: int, 
        bit_width: int, 
        device: str | torch.device = 'cpu', 
        seed: int = 42, 
        norm_correction: bool = True
    ):
        """Khởi tạo hệ thống lượng tử hóa PolarQuant."""
        self.d = d
        self.bit_width = bit_width
        self.device = device
        self.norm_correction = norm_correction
        
        # 1. Khởi tạo mảng mặt nạ dấu ngẫu nhiên (sử dụng chung cho quá trình mã hóa/giải mã)
        self.signs = get_random_signs(d, device=device, seed=seed)
        
        # 2. Khởi tạo từ điển mã (Centroids) được tối ưu hóa cho phân phối chuẩn
        self.centroids = get_optimal_centroids(bit_width, d, device=device)

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Thực hiện quy trình mã hóa (lượng tử hóa) tensor đầu vào.

        Quy trình: 
        1. Trích xuất chuẩn L2 (Norm) -> 2. Chiếu lên mặt cầu đơn vị -> 
        3. Phân tán dữ liệu (Rotate) -> 4. Lượng tử hóa thành chỉ số nguyên (Indices).

        Tham số:
            x (torch.Tensor): Tensor dữ liệu đầu vào. Chiều cuối cùng phải bằng `d`.

        Trả về:
            tuple[torch.Tensor, torch.Tensor]: Bao gồm:
                - indices: Tensor chứa các chỉ số đã bị lượng tử hóa (kiểu số nguyên).
                - norms: Tensor chứa chuẩn L2 nguyên bản (kiểu dấu phẩy động) để bảo toàn biên độ.
        """
        # Bước 1: Trích xuất chuẩn L2 nguyên bản (bảo toàn dưới dạng Float16/Float32)
        norms = torch.norm(x, dim=-1, keepdim=True)
        
        # Bước 2: Chuẩn hóa vector về bề mặt hình cầu đơn vị (Unit Sphere)
        # Cộng thêm epsilon (1e-12) để tránh lỗi số học (chia cho 0) khi vector có độ lớn bằng 0
        x_unit = x / (norms + 1e-12)
        
        # Bước 3: Áp dụng phép xoay không gian (Hadamard Transform + Random Signs)
        # Giúp phân tán trị ngoại lai và làm phẳng phân phối dữ liệu
        y = rotate_tensor(x_unit, self.signs)
        
        # Bước 4: Lượng tử hóa vô hướng thông qua việc ánh xạ giá trị thực thành chỉ số
        indices = quantize_to_indices(y, self.centroids)
        
        return indices, norms

    def dequantize(self, indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """
        Thực hiện quy trình giải mã (giải lượng tử hóa) để khôi phục tensor xấp xỉ.

        Quy trình:
        1. Ánh xạ ngược (Lookup) -> 2. Hiệu chỉnh chuẩn (Norm Correction) -> 
        3. Xoay nghịch đảo (Inverse Rotate) -> 4. Tái tạo biên độ.

        Tham số:
            indices (torch.Tensor): Tensor chứa các chỉ số nguyên từ quá trình mã hóa.
            norms (torch.Tensor): Tensor chứa chuẩn L2 nguyên bản.

        Trả về:
            torch.Tensor: Tensor đã được khôi phục, xấp xỉ với tensor `x` ban đầu.
        """
        # Bước 1: Giải mã từ chỉ số nguyên về các giá trị thực đại diện (Centroids)
        y_hat = dequantize_from_indices(indices, self.centroids)
        
        # Bước 2: Hiệu chỉnh sai số chuẩn L2 (Norm Correction)
        # Sai số lượng tử hóa có thể làm vector bị lệch khỏi mặt cầu đơn vị.
        # Bước này ép buộc vector giải nén nằm chính xác trên mặt cầu đơn vị.
        if self.norm_correction:
            y_hat_norms = torch.norm(y_hat, dim=-1, keepdim=True)
            y_hat = y_hat / (y_hat_norms + 1e-12)
            
        # Bước 3: Áp dụng phép xoay nghịch đảo để khôi phục không gian đặc trưng ban đầu
        x_hat_unit = inverse_rotate_tensor(y_hat, self.signs)
        
        # Bước 4: Tái tạo cấu trúc vector hoàn chỉnh bằng cách nhân lại với chuẩn L2 nguyên bản
        x_hat = x_hat_unit * norms
        
        return x_hat

    def quantize_and_residual(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Thực hiện lượng tử hóa và trả về phần dư (Residual Error).

        Hàm này thiết yếu đối với các hệ thống lượng tử hóa đa giai đoạn (như QJL 
        trong cấu trúc TurboQuant), nơi phần dư của bước này sẽ là đầu vào để nén 
        tiếp ở bước sau nhằm giảm thiểu tối đa sai số lượng tử hóa.

        Tham số:
            x (torch.Tensor): Tensor đầu vào cần nén.

        Trả về:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Bao gồm:
                - indices: Tensor chỉ số mã hóa.
                - norms: Tensor chuẩn L2 gốc.
                - residual: Tensor chứa phần sai số dư (x - x_hat).
        """
        indices, norms = self.quantize(x)
        x_hat = self.dequantize(indices, norms)
        
        # Tính toán sai số lượng tử hóa (Khoảng cách giữa tín hiệu gốc và tín hiệu khôi phục)
        residual = x - x_hat
        
        return indices, norms, residual