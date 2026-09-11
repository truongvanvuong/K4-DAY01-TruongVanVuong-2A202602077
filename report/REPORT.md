# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
    ` class_id : 468, class_name : "cab", rank là 1, score : 0.510915 , taxonomy_name : "ImageNet-1K" `
- Record này mô tả toàn ảnh như thế nào?
    Tác vụ phân loại ảnh gán một nhãn duy nhất (ở đây là "cab") để đại diện cho nội dung chính, chủ thể nổi bật nhất hoặc ngữ cảnh tổng thể của toàn bộ bức ảnh.  
    Nó chỉ cho biết bức ảnh chứa vật thể gì (xe taxi) nhưng hoàn toàn không cung cấp vị trí, tọa độ hay vẽ khung (bounding box) cho biết chiếc xe taxi đó nằm ở đâu trong khung hình.

- Ai định nghĩa class list mà checkpoint có thể dự đoán?
    Danh sách các nhãn (class list) được định nghĩa bởi những nhà nghiên cứu, kỹ sư tạo ra bộ dữ liệu huấn luyện.
    Cụ thể, thông qua giá trị taxonomy_name là "ImageNet-1K", ta có bằng chứng cho thấy danh sách 1000 nhãn này do những người xây dựng bộ dữ liệu ImageNet thiết lập từ trước. 

- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
    Giữ class_id (468): Để hệ thống máy tính dễ dàng đọc, phân loại, tính toán nhanh và tối ưu hóa bộ nhớ (xử lý số luôn nhanh hơn xử lý chuỗi văn bản).  
    Giữ class_name ("cab"): Để thân thiện với con người, giúp người đánh giá (reviewer) đọc vào là hiểu ngay mô hình đang dự đoán vật thể gì mà không cần mở tài liệu tra cứu ID.  
    Giữ taxonomy_name ("ImageNet-1K"): Đóng vai trò như "tên cuốn từ điển". Nó giúp xác định rõ hệ thống phân loại đang dùng, tránh xung đột nhãn khi kết hợp nhiều bộ dữ liệu (vì mã ID 468 ở bộ dữ liệu khác chưa chắc đã là xe taxi). 

- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
    Guideline (hướng dẫn gán nhãn) cần quy định rõ ràng các quy tắc để người gán nhãn (annotator) xử lý sự mơ hồ.
    Cụ thể, cần chỉ rõ: Bắt buộc phải chọn một chủ thể duy nhất làm nhãn chính 
    ví dụ: ưu tiên vật to nhất, vật ở trung tâm hay hệ thống cho phép gán nhiều nhãn cùng lúc (multi-label) cho một bức ảnh. 
    Điều này giúp dữ liệu luôn đồng nhất, không bị tình trạng mỗi nhân viên gán một kiểu.

- Vì sao model score không phải ground truth? 
    Model score (ví dụ: điểm số 0.510915) chỉ là mức độ tự tin (xác suất) bằng con số mà mô hình thuật toán dự đoán, nó hoàn toàn có thể là một dự đoán sai. 
    Ngược lại, Ground truth (nhãn gốc) là sự thật khách quan, là đáp án chuẩn xác 100% do con người trực tiếp kiểm duyệt và gán thủ công. Ground truth được dùng làm "thước đo" để chấm điểm xem cái score mà mô hình đưa ra là đúng hay sai.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
    `class_name: "person",  score: 0.899318 , bbox_xyxy: [385.45, 66.44, 498.02, 348.58] , bbox_width: 112.57, bbox_height: 282.14`
- Diễn giải vị trí box bằng lời:
    Dựa vào tọa độ box của kitchen-001 với nhãn nhận diện (class_name: "person")  đối chiếu lên bức ảnh detection_predictions.jpg, bounding box này khoanh vùng người đầu bếp (person) đang đứng quay lưng lại, nằm ở phía bên phải của bức ảnh. Khung hình bắt đầu từ tọa độ góc trên bên trái (385.45, 66.44) kéo dài xuống góc dưới bên phải (498.02, 348.58), ôm trọn từ phần đỉnh đầu đến gần gót chân của người này.
- So sánh số prediction ở hai threshold:
    Dựa vào file JSON `detection_predictions.json`, model đang sử dụng score_threshold: 0.35 và xuất ra tổng cộng 11 predictions cho sample kitchen (bao gồm person, bowl, potted plant, dining table, oven, spoon).
    Giả sử chúng ta nâng threshold lên 0.50: Dựa vào danh sách điểm số (score), 5 object bao gồm oven (0.489666), spoon (0.487578), person (0.431438), bowl (0.372406), và spoon (0.359146) sẽ bị loại bỏ. Số lượng prediction trên ảnh sẽ giảm từ 11 xuống còn 6 objects. Điều này cũng có thể thấy rõ trên ảnh detection_predictions.jpg khi các box mờ với độ tự tin thấp như "bowl 0.46" hay "cup 0.45" sẽ biến mất nếu threshold bị kéo lên quá cao. 
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
    Về độ bao phủ : Threshold thấp (0.35) giúp tăng độ bao phủ, bắt được cả những vật thể nhỏ, bị mờ hoặc lấp ló phía sau (như các cái bát hoặc muỗng).Nếu đẩy threshold lên cao, model sẽ bỏ sót nhiều vật thể thực tế có tồn tại trong không gian bếp (tăng False Negatives).
    Về khối lượng công việc của Reviewer: Threshold thấp buộc reviewer phải kiểm tra số lượng box lớn hơn rất nhiều, đồng thời tốn thêm thời gian dọn dẹp các "rác" (False Positives) do model đoán nhầm. Ngược lại, threshold cao giúp reviewer rảnh tay hơn, tối ưu tốc độ annotation vì chỉ cần xử lý các box có độ tin cậy rất cao, nhưng chất lượng dataset sẽ không đủ chi tiết.
- Đề xuất một quy tắc box chặt:
    Bounding box phải ôm sát các điểm cực đại (trên, dưới, trái, phải) của phần vật thể CÓ THỂ NHÌN THẤY được trên ảnh, sai số không được vượt quá 2 pixel, và tuyệt đối không bao gồm phần background dư thừa hoặc các vật thể không liên quan lân cận.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
    Trong những bối cảnh lộn xộn, bị che lấp như bức hình `detection_predictions.png`, cần có Guideline cụ thể cho hai trường hợp:
    Độ che khuất (Occlusion): Cần định lượng rõ tỷ lệ che khuất (ví dụ: vật thể bị che khuất > 70% thì có cần vẽ box hay bỏ qua?). Nếu một cái bát (bowl) chỉ lộ ra cái viền nhỏ xíu, reviewer cần biết tiêu chuẩn cụ thể để giữ hay xóa.
    Cắt mép ảnh (Truncation): Box chỉ được vẽ ở phần nhìn thấy được, hay phải dự đoán "vẽ tràn" ra ngoài viền ảnh? (Thông thường nguyên tắc chuẩn là chỉ vẽ sát viền ảnh).
    Quy trình Escalation: Nếu một vật thể bị che khuất đến mức làm biến dạng hình thái (không thể nhận diện bằng mắt nếu tách khỏi ngữ cảnh), reviewer bắt buộc phải escalate cho QA hoặc Admin xác nhận xem có dán nhãn hay không, nhằm đảm bảo model không học nhầm các đặc trưng sai lệch.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
    `instance_id: kitchen-002, class_name: bowl, score: 0.735744, polygon_point_count": 67, [[53.0, 344.0],[52.0, 345.0], [50.0, 345.0], ... ]`
- Polygon bổ sung chi tiết gì so với box?
    Bounding Box (bbox_xyxy): Chỉ cung cấp tọa độ hình chữ nhật bao quanh thô (chứa cả vùng nền xung quanh và không thể hiện được hình dạng thực tế của vật thể).
    Polygon (polygon_xy): Cung cấp đường viền đa giác bám sát theo hình dáng thực tế, đường cong, góc cạnh của đối tượng (ví dụ: miệng và thân cong của cái bát), giúp phân tách chính xác đối tượng khỏi nền và các vật thể xung quanh.  
- `instance_id` dùng để làm gì và không phải loại ID nào?
    Dùng để: Phân biệt duy nhất từng thể hiện (instance) cụ thể của một đối tượng trong cùng một bức ảnh (ví dụ: phân biệt kitchen-002 với kitchen-003 đều là lớp bowl).  
    Không phải là:class_id (mã định danh phân loại của taxonomy như ID 45 cho class bowl).  coco_image_id (mã định danh gốc của bộ dữ liệu COCO như 397133).  
- Đề xuất một quy tắc biên mask:
    Quy tắc "Bám sát biên và không lấn nền" (Edge-Snapping Rule): Ranh giới của mask phải ôm sát pixel thực tế của đối tượng; đối với các vùng biên rõ ràng, đường polygon không được dịch chuyển ra ngoài nền quá 1 pixel, và tại các đoạn chuyển màu mờ (anti-aliasing), mask phải nằm ở tâm của dải chuyển tiếp.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
    Cần guideline và escalation cho các trường hợp:
    Độ mờ (Blur) / Thiếu tương phản: Khi ranh giới vật thể nhập nhèm với nền, cần hướng dẫn cụ thể về việc nên cắt bớt hay giữ lại biên đó.
    Che khuất (Occlusion) & Tiếp xúc (Touching): Khi các vật thể đè lên nhau (ví dụ các bát đĩa hoặc dụng cụ nhà bếp), cần quy định rõ việc nội suy phần bị che khuất hay chỉ vẽ phần nhìn thấy được, đồng thời cần cơ chế escalation để chuyên gia quyết định trong các ca phức tạp.   
## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cho toàn ảnh: `class_id` + `class_name` theo taxonomy ImageNet-1K (ví dụ record hạng 1: `468` / `cab`). Không có box hay polygon. | Sample `traffic`: model chọn `cab` với score ~0.51. Ảnh có thể có nhiều chủ thể; `cab` là tên ImageNet, không tự chứng minh đây là taxi hay chủ thể chính. Score không phải GT. | Chọn đúng một lớp theo guideline (ưu tiên vật lớn nhất / trung tâm, hoặc multi-label nếu guideline cho phép). Không copy top-1 của model. | Ảnh có đúng lớp đã gán không; có xung đột chủ thể không; taxonomy có khớp ImageNet-1K không. Ca nhiều chủ thể hoặc tên lớp mơ hồ thì escalate, không lấy `score` làm đúng. |
| Phát hiện vật thể | Mỗi object một record: lớp COCO 80 + `bbox_xyxy` pixel `[x_min, y_min, x_max, y_max]` (gốc trên-trái). Ví dụ kitchen: `person` `[385.45, 66.44, 498.02, 348.58]`. | Sample `kitchen`: threshold 0.35 giữ 11 prediction; nhiều box score thấp (`oven` ~0.49, `spoon`, `bowl`). Object bị che (bát chỉ lộ viền) hoặc cắt mép: giữ hay bỏ, box chỉ phần nhìn thấy hay vẽ tràn. Threshold lọc prediction, không phải quy tắc GT. | Vẽ box ôm sát phần nhìn thấy, sai số ≤ 2 px, không ôm nền/vật cạnh. Gán đủ object trong phạm vi guideline, kể cả khi model score thấp hoặc model bỏ sót. | Sai lớp, box lệch/rộng, false positive, bỏ sót. Occlusion vượt ngưỡng guideline (ví dụ >70%) hoặc cắt mép không rõ thì escalate. Không dùng threshold như lý do xóa GT. |
| Instance segmentation | Mỗi instance: lớp COCO + `instance_id` + `polygon_xy` (và box kèm theo). Ví dụ `kitchen-002`: `bowl`, 67 điểm polygon. `instance_id` phân biệt hai bát, không phải `class_id`. | Polygon bám hình cái bát, box thì ôm cả nền. Bát/đĩa chồng, biên mờ/anti-aliasing: cắt sát pixel nhìn thấy hay nội suy phần bị che? | Một polygon cho mỗi instance, bám biên ~1 px, không lấn nền; chỉ mask phần nhìn thấy trừ khi guideline cho phép nội suy. Không gộp hai object cùng lớp thành một mask. | Mask lấn nền, hai instance dính nhau, lệch biên, nhầm `instance_id` với `class_id`. Vùng mờ/tiếp xúc/che khuất không đủ thông tin thì escalate cho QA, không tự bịa phần khuất. |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
  Chỉ dùng ba ảnh COCO công khai mà notebook đã cố định. Không tải ảnh cá nhân, khuôn mặt, biển số, dữ liệu khách hàng hoặc dữ liệu nội bộ lên Colab/GitHub công khai. Không ghi họ tên, MSSV, email, số điện thoại hay thông tin cá nhân khác vào `REPORT.md`, JSON, PNG hoặc ZIP.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
  Dừng ngay, không tiếp tục chạy/nộp file đó, rồi báo GV/Lab Coach (kênh hỗ trợ của lớp) kèm mô tả phạm vi sai, không gửi password, token hay mã xác thực.

## 6. Danh sách bằng chứng

- [ ] `classification_predictions.json`
- [ ] `detection_predictions.json`
- [ ] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
