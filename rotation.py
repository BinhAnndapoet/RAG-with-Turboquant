import torch
import math

def fast_walsh_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Thực hiện phép biến đổi Fast Walsh-Hadamard Transform (FWHT).

    Phép biến đổi này giúp phân tán các giá trị ngoại lai (outliers) và làm phẳng 
    phân phối dữ liệu trên toàn bộ chiều cuối cùng của tensor.

    Tham số:
        x (torch.Tensor): Tensor đầu vào cần biến đổi. Kích thước của chiều cuối cùng
            (dim=-1) bắt buộc phải là một lũy thừa của 2.

    Trả về:
        torch.Tensor: Tensor sau khi đã thực hiện phép biến đổi FWHT và được 
            chuẩn hóa (scale), có cùng shape với tensor đầu vào `x`.

    Ngoại lệ:
        AssertionError: Nếu kích thước chiều cuối cùng của `x` không phải là lũy thừa của 2.
    """
    d = x.shape[-1]
    assert (d & (d - 1) == 0) and d > 0, "Kích thước của chiều cuối cùng (dim=-1) phải là lũy thừa của 2."
    
    # Tạo bản sao để tránh thay đổi dữ liệu của tensor gốc
    out = x.clone()
    step = 1
    
    # Thực hiện phép toán Cánh bướm (Butterfly Operations) theo dạng vector hóa
    while step < d:
        shape_orig = out.shape
        # Reshape để ghép cặp các khối phần tử cách nhau khoảng 'step'
        out = out.view(-1, d // (step * 2), 2, step)
        
        # Trích xuất 2 nhánh của biểu đồ cánh bướm
        a, b = out[:, :, 0, :].clone(), out[:, :, 1, :].clone()
        
        # Áp dụng ma trận Hadamard bậc 2: H_2 = [[1, 1], [1, -1]]
        out[:, :, 0, :] = a + b
        out[:, :, 1, :] = a - b
        
        # Khôi phục shape và tăng bước nhảy cho vòng lặp tiếp theo
        out = out.view(shape_orig)
        step *= 2
        
    # Chuẩn hóa với hệ số 1/sqrt(d) để bảo toàn phương sai (L2 norm)
    return out / math.sqrt(d)


def get_random_signs(d: int, device: str | torch.device = 'cpu', dtype: torch.dtype = torch.float32, seed: int = 42) -> torch.Tensor:
    """
    Khởi tạo một vector chứa các giá trị dấu ngẫu nhiên (+1 và -1).

    Trong các hệ thống lượng tử hóa (như TurboQuant), việc kết hợp lật dấu ngẫu nhiên 
    trước khi áp dụng FWHT là bước quan trọng để triệt tiêu các vector cộng hưởng, 
    giúp dữ liệu không bị mất mát thông tin khi nén.

    Tham số:
        d (int): Chiều dài của vector dấu cần tạo.

    Trả về:
        torch.Tensor: Tensor kích thước (d,) chứa các giá trị thuộc tập {-1, 1}.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    
    # Sinh ngẫu nhiên nhị phân {0, 1}, sau đó ánh xạ: 0 -> -1 và 1 -> +1
    signs = torch.randint(0, 2, (d,), generator=gen, device=device, dtype=dtype) * 2 - 1
    
    return signs


def rotate_tensor(x: torch.Tensor, signs: torch.Tensor = None) -> torch.Tensor:
    """
    Thực hiện phép xoay không gian (Rotation) lên tensor.

    Quy trình bao gồm: Nhân phần tử (element-wise) với vector dấu, sau đó áp dụng FWHT.
    Phép toán này giúp "san phẳng" phân phối của dữ liệu, giảm thiểu các giá trị ngoại lai 
    (outliers) trong các tác vụ định lượng hóa (quantization).

    Tham số:
        x (torch.Tensor): Tensor đầu vào cần xoay.
        signs (torch.Tensor, tùy chọn): Vector dấu định trước. Nếu None, hệ thống 
            sẽ tự động khởi tạo vector dấu với seed mặc định.

    Trả về:
        torch.Tensor: Tensor sau khi đã được xoay.
    """
    if signs is None:
        signs = get_random_signs(x.shape[-1], device=x.device, dtype=x.dtype)
        
    # Bước 1: Áp dụng mặt nạ dấu ngẫu nhiên (Element-wise multiplication)
    x_flipped = x * signs
    
    # Bước 2: Phân tán và làm phẳng (Flattening) bằng FWHT
    x_rotated = fast_walsh_hadamard_transform(x_flipped)
    
    return x_rotated

def inverse_rotate_tensor(x_rotated: torch.Tensor, signs: torch.Tensor = None) -> torch.Tensor:
    """
    Thực hiện phép nghịch đảo không gian để khôi phục tensor về trạng thái ban đầu.

    Dựa trên tính chất của ma trận Hadamard (H = H^T), phép biến đổi ngược 
    chính là thực hiện lại FWHT và sau đó nhân lại với cùng một vector dấu.

    Tham số:
        x_rotated (torch.Tensor): Tensor đang ở trạng thái đã bị xoay.
        signs (torch.Tensor, tùy chọn): Vector dấu đã sử dụng ở hàm `rotate_tensor`. 
            Bắt buộc phải khớp với vector dấu lúc mã hóa để giải mã đúng.

    Trả về:
        torch.Tensor: Tensor nguyên bản được khôi phục.
    """
    if signs is None:
        signs = get_random_signs(x_rotated.shape[-1], device=x_rotated.device, dtype=x_rotated.dtype)
        
    # Bước 1: Giải FWHT (Vì FWHT tự nghịch đảo nên FWHT(FWHT(x)) = x)
    x_unrotated = fast_walsh_hadamard_transform(x_rotated)
    
    # Bước 2: Khôi phục lại dấu ban đầu (Vì (+1)*(+1) = 1 và (-1)*(-1) = 1)
    x_restored = x_unrotated * signs
    
    return x_restored