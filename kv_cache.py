import torch
from dataclasses import dataclass
from typing import List, Tuple, Dict
from outlier import OutlierAwareQuantizer, OutlierCompressedKV

@dataclass
class CompressedKVCache:
    """
    Cấu trúc dữ liệu đóng gói toàn bộ trạng thái bộ nhớ đệm KV (KV Cache) đã được nén.
    
    Đại diện cho trạng thái lưu trữ tối ưu trên VRAM, thay thế cho các tensor 4D 
    nguyên bản nhằm giảm thiểu hiện tượng thắt cổ chai băng thông bộ nhớ (Memory Wall) 
    khi suy luận với ngữ cảnh dài (Long-context Inference).

    Thuộc tính:
        k_data (List[List[OutlierCompressedKV]]): Dữ liệu Key đã nén. Cấu trúc danh sách lồng 
            nhau theo thứ tự [layer][head].
        v_data (List[List[OutlierCompressedKV]]): Dữ liệu Value đã nén. Có cùng cấu trúc với `k_data`.
        num_layers (int): Tổng số block biến đổi (Transformer layers).
        num_heads (int): Số lượng cơ chế chú ý (Attention heads) trên mỗi layer.
        seq_len (int): Chiều dài chuỗi token hiện tại (Sequence length).
        head_dim (int): Số chiều đặc trưng của mỗi head (Feature dimension per head).
        k_bits (float): Số bit trung bình được cấp phát để nén mỗi phần tử của Key.
        v_bits (float): Số bit trung bình được cấp phát để nén mỗi phần tử của Value.
    """
    k_data: List[List[OutlierCompressedKV]]
    v_data: List[List[OutlierCompressedKV]]
    num_layers: int
    num_heads: int
    seq_len: int
    head_dim: int
    k_bits: float
    v_bits: float

class KVCacheCompressor:
    """
    Hệ thống mã hóa và giải mã bộ nhớ đệm KV Cache theo từng Head của mô hình LLM.
    
    Hệ thống này sử dụng chiến lược Lượng tử hóa nhận thức Ngoại lai (Outlier-Aware Quantization).
    Đặc biệt, nó hỗ trợ phân bổ bit bất đối xứng (Asymmetric Bit Allocation) giữa Key và Value, 
    do bản chất toán học của toán tử Attention đòi hỏi độ chính xác khác nhau đối với hai ma trận này.
    """

    def __init__(
        self, 
        num_layers: int, 
        num_heads: int, 
        head_dim: int, 
        k_bits: float = 3.5, 
        v_bits: float = 2.5, 
        device: str | torch.device = 'cpu'
    ):
        """
        Khởi tạo hệ thống máy nén KV Cache.

        Tham số:
            num_layers (int): Số lượng layer của mô hình Transformer.
            num_heads (int): Số lượng Attention heads.
            head_dim (int): Kích thước chiều đặc trưng của mỗi head.
            k_bits (float, tùy chọn): Mức bit lượng tử hóa cho ma trận Key. Mặc định là 3.5.
            v_bits (float, tùy chọn): Mức bit lượng tử hóa cho ma trận Value. Mặc định là 2.5.
            device (str | torch.device, tùy chọn): Thiết bị xử lý tensor. Mặc định là 'cpu'.
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.device = device

        # Khởi tạo các bộ nén độc lập cho Key và Value (Chiến lược bất đối xứng)
        # LÝ DO: Ma trận Key (K) tham gia trực tiếp vào việc tính toán điểm số (Attention Scores) 
        # thông qua tích vô hướng, nên các vector cộng hưởng và trị ngoại lai rất nhạy cảm.
        # Do đó, K thường cần số bit cao hơn (VD: 3.5 bits) để bảo toàn năng lực truy xuất (Retrieval).
        self.k_quantizer = OutlierAwareQuantizer(d=head_dim, target_bits=k_bits, device=device)
        
        # Ma trận Value (V) chỉ chịu trách nhiệm tổng hợp thông tin cuối cùng, 
        # nên nó có tính chịu lỗi (fault-tolerance) cao hơn và có thể nén sâu hơn (VD: 2.5 bits).
        self.v_quantizer = OutlierAwareQuantizer(d=head_dim, target_bits=v_bits, device=device)

    def calibrate(self, k_sample: torch.Tensor, v_sample: torch.Tensor) -> None:
        """
        Giai đoạn hiệu chuẩn (Calibration): Quét phân phối dữ liệu mẫu để định vị các kênh ngoại lai.

        Quá trình này chỉ cần chạy một lần trước khi bắt đầu suy luận (hoặc sử dụng 
        tập dữ liệu mồi - prompt prefix) để hệ thống nhận diện các đặc trưng quan trọng.

        Tham số:
            k_sample (torch.Tensor): Tensor chứa mẫu dữ liệu Key, shape `[batch, heads, seq_len, dim]`.
            v_sample (torch.Tensor): Tensor chứa mẫu dữ liệu Value, shape `[batch, heads, seq_len, dim]`.
        """
        # Định dạng lại tensor từ không gian 4D về dạng ma trận 2D `[tổng_số_vector, head_dim]`.
        # Phép transpose(1, 2) đảm bảo các vector thuộc cùng một head được giữ liên tục trong bộ nhớ,
        # giúp quá trình tìm kiếm điểm ngoại lai chính xác hơn theo từng chiều đặc trưng.
        k_flat = k_sample.transpose(1, 2).reshape(-1, self.head_dim)
        v_flat = v_sample.transpose(1, 2).reshape(-1, self.head_dim)
        
        # Kích hoạt quá trình tìm kiếm (fit) vị trí kênh (channel indices) chứa ngoại lai
        self.k_quantizer.fit(k_flat)
        self.v_quantizer.fit(v_flat)

    def compress(self, k_cache: torch.Tensor, v_cache: torch.Tensor) -> CompressedKVCache:
        """
        Nén các tensor bộ nhớ đệm KV nguyên bản thành định dạng tiết kiệm bộ nhớ.

        Hàm sẽ lặp qua từng Layer và từng Head để áp dụng thuật toán lượng tử hóa 
        lên chiều thứ nguyên (dim).

        Tham số:
            k_cache (torch.Tensor): Tensor nguyên bản của Key, shape `[num_layers, num_heads, seq_len, head_dim]`.
            v_cache (torch.Tensor): Tensor nguyên bản của Value, shape `[num_layers, num_heads, seq_len, head_dim]`.

        Trả về:
            CompressedKVCache: Đối tượng dataclass chứa toàn bộ dữ liệu đã được lượng tử hóa.
        """
        seq_len = k_cache.shape[2]
        k_compressed_all = []
        v_compressed_all = []

        for l in range(self.num_layers):
            k_layer = []
            v_layer = []
            for h in range(self.num_heads):
                # Trích xuất toàn bộ các vector thuộc một cơ chế chú ý (Head) cụ thể.
                # Shape đầu vào tại bước này là `[seq_len, head_dim]`.
                k_vecs = k_cache[l, h] 
                v_vecs = v_cache[l, h]
                
                # Thực hiện lượng tử hóa (bảo vệ ngoại lai + nén vô hướng)
                k_layer.append(self.k_quantizer.quantize(k_vecs))
                v_layer.append(self.v_quantizer.quantize(v_vecs))
                
            k_compressed_all.append(k_layer)
            v_compressed_all.append(v_layer)

        # Đóng gói kết quả và metadata vào cấu trúc dữ liệu chung
        return CompressedKVCache(
            k_data=k_compressed_all, v_data=v_compressed_all,
            num_layers=self.num_layers, num_heads=self.num_heads,
            seq_len=seq_len, head_dim=self.head_dim,
            k_bits=self.k_bits, v_bits=self.v_bits
        )

    def decompress(self, compressed: CompressedKVCache) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Giải nén và khôi phục (xấp xỉ) các tensor KV 4D để sẵn sàng cho tính toán Attention.

        Tham số:
            compressed (CompressedKVCache): Đối tượng chứa trạng thái KV đã nén.

        Trả về:
            Tuple[torch.Tensor, torch.Tensor]: Bao gồm:
                - Khối tensor Key (k_out) shape `[num_layers, num_heads, seq_len, head_dim]`.
                - Khối tensor Value (v_out) shape `[num_layers, num_heads, seq_len, head_dim]`.
        """
        # Khởi tạo vùng nhớ liên tục (contiguous memory) cho các tensor đầu ra
        shape = (compressed.num_layers, compressed.num_heads, compressed.seq_len, compressed.head_dim)
        k_out = torch.zeros(shape, device=self.device)
        v_out = torch.zeros(shape, device=self.device)

        # Khôi phục dữ liệu theo từng Layer và từng Head
        for l in range(compressed.num_layers):
            for h in range(compressed.num_heads):
                # Giải lượng tử hóa lấy lại giá trị thực (Float16/Float32)
                k_out[l, h] = self.k_quantizer.dequantize(compressed.k_data[l][h])
                v_out[l, h] = self.v_quantizer.dequantize(compressed.v_data[l][h])
                
        return k_out, v_out

    def memory_stats(self, seq_len: int) -> Dict[str, float]:
        """
        Đánh giá lý thuyết về mức độ tiết kiệm bộ nhớ (VRAM) của hệ thống.

        Tính toán dựa trên giả định mô hình gốc lưu trữ KV ở định dạng Float16 (2 bytes/giá trị)
        và so sánh với dung lượng trung bình khi áp dụng số bit cấu hình (bao gồm cả chi phí lưu metadata).

        Tham số:
            seq_len (int): Chiều dài chuỗi ngữ cảnh cần đánh giá.

        Trả về:
            Dict[str, float]: Bảng thống kê bao gồm kích thước gốc (MB), kích thước nén (MB) 
                và tỷ lệ nén (Compression Ratio).
        """
        # Tính tổng số lượng vector trong toàn bộ KV Cache
        total_vectors = self.num_layers * self.num_heads * seq_len
        
        # Mức tiêu thụ nguyên thủy với Float16 (16-bit = 2 bytes cho mỗi giá trị)
        original_bytes = total_vectors * self.head_dim * 2 * 2
        
        # Tính mức tiêu thụ sau khi nén:
        # - Payload: Dữ liệu bị lượng tử hóa (head_dim * bit_rate / 8 byte)
        # - Metadata: Chi phí lưu trữ chuẩn L2 (Norm) dùng Float32 (4 bytes cho mỗi vector)
        k_bytes = total_vectors * ((self.head_dim * self.k_bits / 8) + 4)
        v_bytes = total_vectors * ((self.head_dim * self.v_bits / 8) + 4)
        compressed_bytes = k_bytes + v_bytes
        
        return {
            "original_mb": original_bytes / (1024**2),
            "compressed_mb": compressed_bytes / (1024**2),
            "ratio": original_bytes / compressed_bytes
        }