# Toàn bộ công thức thuật toán (phiên bản v3 — đã đồng bộ với code)

## PHẦN 0: Ký hiệu và tham số đầu vào

| Ký hiệu | Ý nghĩa |
|---|---|
| $N = \{0, 1, ..., n\}$ | Tập điểm; $0$ = kho trung tâm, $1..n$ = khách hàng |
| $D = \{1, ..., 7\}$ | Các ngày trong tuần (1 = Thứ 2, ..., 7 = Chủ nhật) |
| $C_d$ | Tập khách hàng chưa giao vào đầu ngày $d$ |
| $(x_i, y_i)$ | Tọa độ khách hàng $i$ (km) |
| $v = 50$ km/h | Tốc độ tối đa của xe |
| $s_i$ | Thời gian phục vụ tại khách hàng $i$ (lấy từ dữ liệu `service_time`) |
| $TW_i^d$ | Tập các khung giờ $[a, b]$ khách $i$ có thể nhận hàng vào ngày $d$ |
| $\tau$ | Ngưỡng trần thời gian chờ tối đa (phút), tham số cần hiệu chỉnh |
| $w_1, w_2, w_3$ | Trọng số của 3 tiêu chí chấm điểm, $w_1+w_2+w_3=1$ |
| $\delta, \lambda$ | Hệ số độ dốc của 2 hàm sigmoid |
| $\alpha, \beta$ | Trọng số hàm chi phí Cost(σ) |

**Khoảng cách và thời gian di chuyển:**

$$d_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2} \tag{1}$$

$$t_{ij} = \frac{d_{ij}}{v} \tag{2}$$

*Giải thích:* khoảng cách Euclid quy đổi trực tiếp sang thời gian di chuyển theo tốc độ tối đa cho phép trong đô thị (50km/h).

---

## PHẦN 1: Công thức thời gian đến / rời tại mỗi điểm trong route

Cho route $\sigma = (\sigma_1, ..., \sigma_k)$ trong ngày $d$, $\sigma_0 = 0$ (kho), xét điểm $\sigma_p$:

**Thời gian đến:**

$$A_{\sigma_p} = L_{\sigma_{p-1}} + t_{\sigma_{p-1}, \sigma_p} \tag{3}$$

**Chọn khung giờ khả thi sớm nhất trong $TW_{\sigma_p}^d$:**

$$w^* = \arg\min_{[a,b] \in TW_{\sigma_p}^d} \{ b : b \geq A_{\sigma_p} \} \tag{4}$$

Nếu không tồn tại $w^*$ → khách hàng **không khả thi trong ngày này**, đưa vào diện xét dời ngày.

**Thời gian bắt đầu phục vụ (chờ nếu đến sớm):**

$$S_{\sigma_p} = \max(A_{\sigma_p}, a^*) \tag{5}$$

**Thời gian chờ:**

$$\text{Wait}_{\sigma_p} = \max(0,\ a^* - A_{\sigma_p}) \tag{6}$$

**Thời gian rời điểm:**

$$L_{\sigma_p} = S_{\sigma_p} + s_{\sigma_p} \tag{7}$$

**Điều kiện khả thi toàn route** (mọi $p$):

$$S_{\sigma_p} \leq b^*_{\sigma_p} \tag{8}$$

**Trường hợp đặc biệt — route rỗng ($k=0$):**

$$k = 0 \implies \text{Dist}(\sigma) = 0,\quad \text{TotalWait}(\sigma) = 0,\quad \text{route luôn khả thi} \tag{8'}$$

*Giải thích:* đây là bộ công thức lõi, mô phỏng chính xác cơ chế "đến sớm phải chờ, đến trễ quá giờ thì không giao được" theo đúng mô tả đề bài. Trường hợp (8') cần được xử lý riêng trong code (return sớm trước khi cộng khoảng cách về kho), tránh việc mặc định gán `current_loc = depot` rồi tính khoảng cách `dist(depot, depot)` không tồn tại trong ma trận khoảng cách.

---

## PHẦN 2: Chuẩn hóa 3 tiêu chí chấm điểm

Tại bước đang đứng ở điểm $u$, xét tập ứng viên $\text{candidates}$.

### 2.1. Độ gấp về thời gian (Slack) — chuẩn hóa bằng sigmoid

$$\text{Slack}_i = b^*_i - S_i \tag{9}$$

$$\mu = \text{mean}\{\text{Slack}_j : j \in \text{candidates},\ \text{Slack}_j \geq 0\} \tag{10}$$

$$g_i = \frac{1}{1 + \exp\big(\delta \cdot (\text{Slack}_i - \mu)\big)} \tag{11}$$

*Giải thích:* Slack càng nhỏ (khách sắp hết giờ nhận hàng) → $g_i$ càng gần 1 → được ưu tiên chọn trước. $\mu$ tính động theo tập ứng viên hiện tại (không cố định toàn cục), giúp thang đo luôn phản ánh đúng bối cảnh còn lại tại từng bước.

### 2.2. Thời gian chờ (Wait) — chuẩn hóa bằng sigmoid

$$\nu = \text{mean}\{\text{Wait}_j : j \in \text{candidates},\ \text{Wait}_j > 0\} \tag{12}$$

$$h_i = \frac{1}{1 + \exp\big(\lambda \cdot (\text{Wait}_i - \nu)\big)} \tag{13}$$

*Giải thích:* Wait càng nhỏ → $h_i$ càng gần 1 → ưu tiên khách không bắt xe phải chờ lâu, giảm lãng phí thời gian vận hành.

### 2.3. Độ gần — chuẩn hóa min-max

$$d_{\min} = \min_{j \in \text{candidates}} d_{u,j}, \qquad d_{\max} = \max_{j \in \text{candidates}} d_{u,j} \tag{14}$$

$$p_i = 1 - \frac{d_{u,i} - d_{\min}}{d_{\max} - d_{\min} + \epsilon}, \qquad \epsilon = 10^{-6} \tag{15}$$

*Giải thích:* Thay vì $p_i = \frac{1}{1+d_{u,i}}$ (bị co về gần 0 vì đơn vị km), công thức này đưa $p_i$ về đúng thang $[0,1]$ giống $g_i, h_i$, đảm bảo trọng số $w_3$ thực sự có tác dụng. Nếu $|\text{candidates}|=1$, quy ước $p_i = 1$.

---

## PHẦN 3: Ngưỡng chờ động và cơ chế 2 tầng chọn ứng viên

**Ngưỡng chờ tối đa tại vị trí $u$:**

$$\text{MaxWait}(u) = \min\Big(\tau,\ \max_{j \in \text{candidates}} t_{u,j}\Big) \tag{16}$$

*Giải thích:* ngưỡng không cố định mà co giãn theo "còn bao xa để tranh thủ đi giao món khác" — nếu candidates còn khách ở xa (đi mất nhiều thời gian), ngưỡng chờ cho phép cao hơn vì có thể tranh thủ; ngược lại nếu candidates đều gần, ngưỡng thấp hơn để tránh chờ vô ích.

**Tập ứng viên khả dụng ngay (tầng 1):**

$$\text{candidates}_{\text{active}} = \{ i \in \text{candidates} : \text{Wait}_i \leq \text{MaxWait}(u) \} \tag{17}$$

**Quy tắc chọn ứng viên tiếp theo:**

$$
i^* = \begin{cases}
\displaystyle\arg\max_{i \in \text{candidates}_{\text{active}}} \text{Score}(i) & \text{nếu } \text{candidates}_{\text{active}} \neq \emptyset \\[2mm]
\displaystyle\arg\min_{i \in \text{candidates}} \text{Wait}_i & \text{nếu } \text{candidates}_{\text{active}} = \emptyset
\end{cases} \tag{18}$$

**Quy tắc phá vỡ thế hòa (tie-breaking):**

$$\text{Nếu } \exists\ i_1 \neq i_2 : \text{Score}(i_1) = \text{Score}(i_2) = \max(\cdot) \implies \text{chọn } i^* = \min(\text{ID}(i_1), \text{ID}(i_2)) \tag{18'}$$

*Giải thích:* nếu có ứng viên chờ ít → chọn theo điểm số tổng hợp như bình thường (tầng 1). Nếu **mọi** ứng viên còn lại đều buộc phải chờ lâu (tầng 2) → chấp nhận chọn khách chờ ít nhất trong số đó, đảm bảo thuật toán luôn tiến triển. Công thức (18') **bắt buộc phải cài đặt tường minh** trong code (vd bằng cách luôn sắp xếp candidates theo ID trước khi duyệt/so sánh) — nếu không, kết quả sẽ phụ thuộc vào thứ tự duyệt ngẫu nhiên của cấu trúc dữ liệu (tập hợp), khiến thuật toán **không tái lập được** giữa các lần chạy khác nhau.

---

## PHẦN 4: Công thức chấm điểm tổng hợp

$$\text{Score}(i) = w_1 \cdot g_i + w_2 \cdot h_i + w_3 \cdot p_i, \qquad w_1+w_2+w_3=1 \tag{19}$$

*Giải thích:* $w_1$ nên lớn nhất (ưu tiên tránh trễ giờ/mất đơn — hậu quả nặng nhất), $w_2, w_3$ điều chỉnh theo mức độ đội thi muốn tối ưu vận hành (giảm chờ) hay giảm quãng đường.

---

## PHẦN 5: Hiệu chỉnh tham số $\delta, \lambda, \tau$ bằng Grid Search

> **Lưu ý phạm vi:** Phần này hiệu chỉnh $\delta, \lambda, \tau$. Trọng số $w_1, w_2, w_3$ (công thức 19) và $\alpha, \beta$ (công thức 29) **không** nằm trong không gian tìm kiếm của Grid Search — đây là các tham số được **chọn chủ quan và giải thích lý do** (theo đúng yêu cầu "nêu rõ tiêu chí nào được ưu tiên hơn" của đề bài), không phải tham số học từ dữ liệu. Nếu muốn mở rộng grid search sang cả $w_1,w_2,w_3,\alpha,\beta$, cần đánh giá lại chi phí tính toán vì không gian tổ hợp sẽ tăng theo cấp số nhân.

**Tách tập holdout để đánh giá (không phải "train" theo nghĩa học máy):**

$$\text{val} = \text{holdout\_sample}(\text{customers},\ n_{\text{val}}) \tag{20}$$

$$n_{\text{val}} = \max\big(n_{\min},\ \lceil \gamma \cdot n \rceil\big) \tag{20'}$$

trong đó $\gamma = 0.3$ (tỷ lệ lấy mẫu), $n_{\min} = 30$ (số khách hàng tối thiểu để tập validation có ý nghĩa thống kê), $n$ = tổng số khách hàng.

*Giải thích:* gọi là "holdout" thay vì "train/val split" vì thuật toán không có bước huấn luyện tham số nào — grid search chỉ đơn giản là chạy thử toàn bộ thuật toán (đã cố định) với các tổ hợp $(\delta,\lambda,\tau)$ khác nhau trên **cùng một tập** `val`, rồi so sánh kết quả. Cần ràng buộc $n_{\min}$ vì nếu $n$ nhỏ, tập lấy mẫu 30% có thể chỉ còn vài khách hàng — không đủ để phân biệt hiệu ứng của tham số (dễ dẫn đến hiện tượng "bão hòa": mọi tổ hợp đều cho cùng 1 kết quả).

**Không gian tìm kiếm:**

$$\delta, \lambda \in \{0.01,\ 0.03,\ 0.05,\ 0.1\}, \qquad \tau \in \{30, 60, 90, 120\}\text{ phút} \tag{21}$$

*(Có thể mở rộng thêm $\{0.3, 0.5\}$ cho $\delta,\lambda$ nếu muốn khảo sát vùng sigmoid dốc hơn — nhưng cần cân nhắc thời gian chạy tăng theo tổ hợp.)*

**Hàm mục tiêu (Objective) — đã bao gồm chuẩn hóa nhất quán với Cost(σ):**

$$\text{Objective}(\delta,\lambda,\tau) = M \cdot |\text{Unfulfilled}(\delta,\lambda,\tau)| \;+\; \alpha \cdot \frac{\text{Dist}_{\text{week}}(\delta,\lambda,\tau)}{\text{REF\_DIST}} \;+\; \beta \cdot \frac{\text{Wait}_{\text{week}}(\delta,\lambda,\tau)}{\text{REF\_WAIT}} \tag{21'}$$

trong đó:
- $M$: hằng số phạt rất lớn (vd $M = 10000$), đảm bảo **không đơn hàng nào bị hy sinh chỉ để giảm quãng đường** — giữ đúng thứ tự ưu tiên đã đề xuất ở công thức (19)
- $\alpha, \beta$: **dùng lại đúng trọng số** đã định nghĩa ở công thức (29)
- $\text{REF\_DIST}, \text{REF\_WAIT}$: **hằng số chuẩn hóa giống hệt** dùng trong  Local Search (Phần 7) — bắt buộc phải nhất quán, nếu không Grid Search sẽ tối ưu một hàm mục tiêu khác với hàm mà Local Search thực sự tối thiểu hóa, dẫn đến chọn sai tham số khi độ lớn của Dist và Wait lệch nhau đáng kể
- $\text{Dist}_{\text{week}}, \text{Wait}_{\text{week}}$: tính theo công thức (32), (33), **sau khi đã áp dụng Local Search** (không phải route thô ban đầu)

**Chọn tham số tối ưu:**

$$(\delta^*, \lambda^*, \tau^*) = \arg\min_{\delta,\lambda,\tau} \text{Objective}(\delta,\lambda,\tau) \tag{22}$$

**Nếu có nhiều tổ hợp cùng đạt Objective tối ưu (hoặc rất gần nhau), ưu tiên chọn tổ hợp đơn giản nhất (giá trị $\tau$ nhỏ nhất):**

$$(\delta^*,\lambda^*,\tau^*) = \arg\min_{\tau} \Big\{ (\delta,\lambda,\tau) : \text{Objective}(\delta,\lambda,\tau) = \min(\text{Objective}) \Big\} \tag{22'}$$

*Giải thích:* nguyên tắc Occam's razor — nếu $\tau=60$ và $\tau=120$ cho cùng chất lượng, chọn $\tau=60$ vì ít rủi ro hơn khi áp dụng cho dữ liệu mới (ngưỡng chờ nhỏ hơn = ít khả năng xe phải đứng chờ lâu không cần thiết trong tình huống thực tế khác với tập validation).

**Đánh giá độ nhạy — tách 2 chỉ số riêng biệt (không gộp chung):**

$$\text{FeasibilityRate} = \frac{\big|\{(\delta,\lambda,\tau) \in \text{grid} : \text{Unfulfilled}(\delta,\lambda,\tau) = 0\}\big|}{|\text{grid}|} \tag{23a}$$

$$\text{Sensitivity}_{\text{quality}} = \frac{\displaystyle\max_{(\delta,\lambda,\tau) \in \text{feasible}} \text{Objective} - \min_{(\delta,\lambda,\tau) \in \text{feasible}} \text{Objective}}{\displaystyle\min_{(\delta,\lambda,\tau) \in \text{feasible}} \text{Objective}} \tag{23b}$$

trong đó $\text{feasible} = \{(\delta,\lambda,\tau) \in \text{grid} : \text{Unfulfilled}(\delta,\lambda,\tau) = 0\}$ — **chỉ tính trên tập con đã đảm bảo giao hết đơn**, không tính trên toàn bộ không gian tham số.

*Giải thích quan trọng:* nếu gộp chung 1 công thức Sensitivity duy nhất trên toàn bộ grid (bao gồm cả các tổ hợp có `Unfulfilled > 0`), số hạng phạt $M \cdot |\text{Unfulfilled}|$ sẽ áp đảo hoàn toàn phần còn lại của Objective, khiến Sensitivity tính ra rất lớn (hàng nghìn) nhưng **không phản ánh đúng** độ ổn định của chất lượng route — mà chỉ phản ánh sự chênh lệch nhị phân "có/không giao hết đơn" giữa các tham số. Tách thành (23a) và (23b) mới cho kết quả diễn giải được: (23a) trả lời "bao nhiêu % không gian tham số là an toàn", (23b) trả lời "trong vùng an toàn đó, chất lượng route có ổn định không".

---

## PHẦN 6: Cơ chế dời ngày

$$C_{d+1} = C_d \setminus \text{Served}_d \tag{24}$$

**Điều kiện được xét dời sang ngày $d'>d$:**

$$TW_i^{d'} \neq \emptyset \tag{25}$$

**Điều kiện không hoàn thành (hết tuần):**

$$\text{Unfulfilled} = C_7 \setminus \text{Served}_7 \tag{26}$$

*Giải thích:* sau ngày Chủ nhật ($d=7$), không còn ngày nào để dời tiếp, khớp đúng định nghĩa "không hoàn thành" trong đề bài.

---

## PHẦN 7: Hàm chi phí (Local Search + đánh giá)

**Tổng quãng đường trong ngày $d$ (gồm cả đoạn về kho, áp dụng công thức 8' nếu route rỗng):**

$$\text{Dist}(\sigma) = \begin{cases} 0 & \text{nếu } k=0 \\ \displaystyle\sum_{p=1}^{k} d_{\sigma_{p-1}, \sigma_p} + d_{\sigma_k, 0} & \text{nếu } k \geq 1 \end{cases} \tag{27}$$

**Tổng thời gian chờ trong ngày $d$:**

$$\text{TotalWait}(\sigma) = \sum_{p=1}^{k} \text{Wait}_{\sigma_p} \tag{28}$$

**Hàm chi phí tổng hợp (đã chuẩn hóa bằng hằng số tham chiếu):**

$$\text{Cost}(\sigma) = \alpha \cdot \frac{\text{Dist}(\sigma)}{\text{REF\_DIST}} + \beta \cdot \frac{\text{TotalWait}(\sigma)}{\text{REF\_WAIT}} \tag{29}$$

trong đó $\text{REF\_DIST}, \text{REF\_WAIT}$ là 2 hằng số chuẩn hóa cố định (vd $100$ km và $200$ phút) — **cùng giá trị phải được dùng lại ở công thức (21') của Phần 5** để đảm bảo tính nhất quán giữa bước hiệu chỉnh tham số và bước tối ưu route thực tế.

**Điều kiện chấp nhận 1 phép biến đổi (Or-opt / 2-opt / swap) trong local search:**

$$\sigma' \text{ được chấp nhận} \iff \text{Feasible}(\sigma')\ \land\ \text{Cost}(\sigma') < \text{Cost}(\sigma) \tag{30}$$

*Giải thích:* $\alpha, \beta$ là trọng số do đội thi chọn, thể hiện ưu tiên giữa "tiết kiệm quãng đường" và "giảm thời gian chờ khách hàng" — cần giải thích rõ lý do lựa chọn trong bài làm (đây chính là Yêu cầu 2 của đề).

---

## PHẦN 8: Các chỉ số đánh giá cuối cùng (dùng cho so sánh với baseline)

**Tỷ lệ hoàn thành đơn hàng (chỉ số quan trọng nhất):**

$$R_{\text{fulfill}} = \frac{|N\setminus\{0\}| - |\text{Unfulfilled}|}{|N\setminus\{0\}|} \tag{31}$$

**Tổng quãng đường cả tuần:**

$$\text{Dist}_{\text{week}} = \sum_{d=1}^{7} \text{Dist}(\sigma_d) \tag{32}$$

**Tổng thời gian chờ cả tuần:**

$$\text{Wait}_{\text{week}} = \sum_{d=1}^{7} \text{TotalWait}(\sigma_d) \tag{33}$$

**Số lượt đơn hàng bị dời ngày (đo độ "mượt" của lịch):**

$$N_{\text{deferred}} = \sum_{d=1}^{6} |C_{d+1}| \tag{34}$$

*Giải thích:* nếu một đơn bị dời 2 lần liên tiếp thì được tính 2 lần trong $N_{\text{deferred}}$, phản ánh đúng mức độ "phiền" mà khách hàng gặp phải. Lưu ý chỉ số này có xu hướng bị "phóng đại" đối với các khách hàng cuối cùng rơi vào `Unfulfilled` (vì họ góp mặt trong backlog liên tục từ ngày 1 đến ngày 6) — nên khi trình bày, cần diễn giải rõ mối liên hệ giữa (34) và (26) thay vì coi đây là 2 chỉ số hoàn toàn độc lập.

---

## Sơ đồ tổng thể liên kết các công thức

```
Với mỗi ngày d = 1..7 (còn khách hàng pending):

  Với mỗi bước xây route (đứng tại điểm u):
    (3)-(8'):  tính A_i, chọn w*, tính S_i, Wait_i, L_i, kiểm tra khả thi
               (xử lý riêng route rỗng theo 8')
    (9)-(15):  tính Slack_i, g_i, h_i, p_i (đã chuẩn hóa)
    (16)-(17): tính MaxWait(u), lọc candidates_active
    (18)-(18'): chọn i* theo quy tắc 2 tầng + tie-breaking theo ID
    (19):      Score(i) = w1*g_i + w2*h_i + w3*p_i

  Sau khi xây xong route ban đầu:
    (27)-(30): Local search cải thiện bằng Or-opt/2-opt/swap, minimize Cost(σ)
               (dùng REF_DIST, REF_WAIT để chuẩn hóa)

  (24)-(25): C_{d+1} = C_d \ Served_d (kiểm tra TW còn hợp lệ không)

Sau ngày 7:
  (26): Unfulfilled = C_7 \ Served_7

Hiệu chỉnh tham số (chạy trước khi áp dụng chính thức, KHÔNG bao gồm w1,w2,w3):
  (20)-(20'): tách tập holdout (không phải train/val theo nghĩa ML)
  (21)-(21'): grid search (δ, λ, τ), Objective dùng CÙNG REF_DIST/REF_WAIT với (29)
  (22)-(22'): chọn tham số tối ưu, ưu tiên đơn giản nếu có nhiều lựa chọn ngang nhau
  (23a)-(23b): tách FeasibilityRate và Sensitivity_quality (không gộp chung)

Đánh giá cuối:
  (31)-(34): R_fulfill, Dist_week, Wait_week, N_deferred
```
