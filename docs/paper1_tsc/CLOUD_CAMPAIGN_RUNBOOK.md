# Cloud Campaign Runbook — W5-9 (Paper 1 MAPPO TSC)

> Máy: GCP `c2d-standard-56` (56 vCPU), Ubuntu 22.04, on-demand, CPU-only.
> Code khóa tại commit **`3e23551`**. Deadline: **30/6**.
> Source of truth lệnh gốc: [`MASTER_PLAN.md`](MASTER_PLAN.md) §2. File này = bản
> thực thi đã chỉnh worker cho 56 vCPU + thread-pinning `OMP_NUM_THREADS=1`
> (mỗi job ~1 nhân ⇒ mỗi stage chạy gọn 1 wave).

---

## 0. Đã khóa sẵn (đi theo `git clone`, KHÔNG cần làm lại trên VM)

| Hạng mục | Trạng thái |
|---|---|
| Noise calib `class_flip_rate=0.302` (measured) | ✅ trong `configs/noise_config.json` |
| OOD demands (4×2 mạng), moto50/73/90, lanebased cfg | ✅ |
| Reward schema 1.2.0 + presets (gồm `no_delay`/`no_low_speed`) | ✅ |
| Lane-group topology n3_grid (16 TLS) | ✅ pilot PASS |
| `pytest 60/60`, `check_obs_match` structural 6/6 | ✅ (local) |
| Legacy ckpt cách ly, repo sạch | ✅ |

→ Trên VM chỉ cần xác nhận **môi trường SUMO** (Phase B), không phải làm lại các mục trên.

---

## PHASE A — Setup môi trường (1 lần, ~15 phút)

```bash
set -e
sudo apt update && sudo apt install -y python3.10 python3.10-venv git tmux htop
sudo add-apt-repository -y ppa:sumo/stable && sudo apt update
sudo apt install -y sumo sumo-tools
git clone https://github.com/tronghia256-EdgeAI/Sim2Real-MAPPO-Traffic.git
cd ~/Sim2Real-MAPPO-Traffic
python3.10 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
python -c "import libsumo, traci" 2>/dev/null || pip install libsumo traci sumolib
cat >> ~/.bashrc <<'RC'
export SUMO_HOME=/usr/share/sumo
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
source ~/Sim2Real-MAPPO-Traffic/.venv/bin/activate
RC
source ~/.bashrc
```

## PHASE B — VERIFY GATE (BẮT BUỘC PASS trước khi đốt compute)

```bash
cd ~/Sim2Real-MAPPO-Traffic
git fetch && git status -sb              # PHẢI: "up to date with origin/master", working tree sạch
# (code/config khóa từ 3e23551; các commit docs sau đó đẩy HEAD lên nhưng KHÔNG đổi training)
python -c "import libsumo; print('libsumo OK')"
echo "SUMO_HOME=$SUMO_HOME OMP=$OMP_NUM_THREADS nproc=$(nproc)"   # /usr/share/sumo, 1, 56
df -h ~ | tail -1                        # còn ≥ vài chục GB
python scripts/check_obs_match.py --sumo # PASS + cosine ~1.0000  ← gate quan trọng nhất
python scripts/run_pilot.py --network n3_grid --total-timesteps 2000   # in PILOT SUMMARY, no crash
```
❌ Bất kỳ dòng nào fail → DỪNG, gửi log. ✅ Tất cả pass → sang Phase C.

> **⚡ CHIẾN LƯỢC: chạy ĐỒNG THỜI cả 3 stage ngay Day 1** (không chờ Stage 1 xong).
> 56 vCPU + OMP=1 ⇒ mỗi job ~1 nhân. Phân bổ **30 (Stage 1) + 17 (2a) + 9 (2b) = 56 worker**
> trong 3 cửa sổ tmux ⇒ máy luôn 100% bận ⇒ **toàn bộ training ~2.5 ngày** (thay vì ~5
> ngày kiểu tuần tự). Đây là cách giữ chi phí trong **$300** (xem mục Chi phí). Stage 2 là
> các run training độc lập, KHÔNG phụ thuộc Stage 1 nên chạy song song được.

## PHASE C — Stage 1: Main campaign (30 job) — cửa sổ tmux #1

```bash
tmux new -s s1
cd ~/Sim2Real-MAPPO-Traffic
python scripts/parallel_launcher.py \
  --networks n2_corridor n3_grid --algos mappo ippo --obs-modes proxy privileged \
  --seeds 42 123 456 789 1337 --total-timesteps 500000 \
  --max-workers 30 --stagger 45 --campaign-id main_05M \
  2>&1 | tee campaign_main.log
```
- 30 job = 2 nets × {mappo-proxy, mappo-priv, ippo-proxy} × 5 seeds (ippo-priv tự skip).
- `--max-workers 30` (chừa nhân cho 2a/2b chạy song song). 1 wave ≈ **~54h (~2.3 ngày)**.
- Thấy job khởi động → **`Ctrl-b` rồi `d`** để detach → mở Phase E NGAY (không chờ).

## PHASE D — Monitor (mở SSH bất cứ lúc nào)

```bash
tmux attach -t s1                                              # xem trực tiếp (Ctrl-b d để thoát)
htop                                                           # ~30 nhân bận
find results/paper1_mappo/main_05M -name bench.json | wc -l    # tiến độ ?/30
grep -ic error campaign_main.log                               # nên = 0
```

## PHASE E — Stage 2: Ablations (chạy ĐỒNG THỜI với Stage 1 — KHÔNG chờ)

> ⚠️ `--reward-presets` × `--obs-ablations` là CROSS-PRODUCT ⇒ 2 lệnh RIÊNG, 2 cửa sổ tmux.
> Worker: 2a=17 + 2b=9 = 26, cộng Stage 1=30 ⇒ **tổng 56 = đúng số nhân** (OMP=1, không oversubscribe).
> Mở 2 cửa sổ này NGAY sau khi launch Phase C (đừng đợi Stage 1 xong).

```bash
# cửa sổ #2:  tmux new -s s2a   (reward ablations, 18 job)
python scripts/parallel_launcher.py --networks n3_grid --algos mappo --obs-modes proxy \
  --reward-presets no_pressure no_throughput queue_only unsigned_pressure mean_then_square no_delay \
  --seeds 42 123 456 --total-timesteps 500000 \
  --max-workers 17 --stagger 45 --campaign-id ablation_reward_05M 2>&1 | tee camp_2a.log

# cửa sổ #3:  tmux new -s s2b   (obs ablations, 9 job)
python scripts/parallel_launcher.py --networks n3_grid --algos mappo --obs-modes proxy \
  --obs-ablations no_class_shares no_pressure_feature lane_truncated \
  --seeds 42 123 456 --total-timesteps 500000 \
  --max-workers 9 --stagger 45 --campaign-id ablation_obs_05M 2>&1 | tee camp_2b.log
```
Cả 3 stage chạy cùng lúc ⇒ **toàn bộ ~2.5 ngày**. (Tùy chọn thêm `no_low_speed` vào 2a ⇒ 21 job;
lúc đó để Stage 1 xong rồi mới chạy phần dư để khỏi quá 56 nhân.)

## PHASE F — Stage 3: Sublane cross-eval (chạy khi job 3 stage trên bắt đầu nhả nhân)

```bash
python experiment/runners/train_ppo.py --mode train --algo mappo --obs-mode proxy \
  --sumo-cfg    sumo_configs/networks/n3_grid/sumo_config_lanebased.sumocfg \
  --lane-groups sumo_configs/networks/n3_grid/lane_groups.json \
  --seed 42 --total-timesteps 500000 --ckpt-dir results/paper1_mappo/sublane_lanebased
# sau đó cross-eval 2 chiều (lane↔sublane) bằng eval_compare.py → TABLE-9
```

## PHASE G — Dựng bảng/figure (sau khi đủ data)

```bash
# baselines (không cần training) — có thể chạy từ Day 2
python experiment/baselines/webster.py  --network n3_grid --seed 42
python experiment/baselines/actuated.py --network n3_grid --seed 42

# turnkey: eval → LaTeX tables → figures (tự tìm campaign mới nhất)
python scripts/build_paper_artifacts.py
python scripts/build_paper_artifacts.py --noise      # FIG-6 noise sweep (5-seed, ~4-5h)
# thành phần lẻ nếu cần:
python experiment/runners/make_tables.py --auto      # figures/tables/*.tex
python experiment/plots/campaign_figures.py --all
```
Chi tiết thứ tự + ánh xạ paper-item: [`POST_TRAINING_RUNBOOK.md`](POST_TRAINING_RUNBOOK.md).

## PHASE H — Lấy kết quả + TẮT MÁY (ngừng tính tiền)

```bash
# từ máy local (PowerShell):
scp -r ubuntu@<IP>:~/Sim2Real-MAPPO-Traffic/results ./results_cloud
scp -r ubuntu@<IP>:~/Sim2Real-MAPPO-Traffic/figures ./figures_cloud
# hoặc commit+push ngay trên VM (results/figures được track)
```
- ⚠️ **GCP Console → Compute Engine → chọn VM → DELETE** (Stop không đủ — disk vẫn tính phí).
- Tải results/figures về TRƯỚC khi delete.

---

## Timeline kỳ vọng (56 vCPU, ĐỒNG THỜI, deadline 30/6)

| | Việc | Wall-clock | Xong ~ |
|---|---|---|---|
| Phase A+B | setup + verify | ~30 phút | ngày 0 |
| Phase C+E (+F) | **toàn bộ training 59 job, đồng thời 30+17+9 worker** | **~2.5 ngày** | ngày 2–3 |
| Phase G | eval + tables + figures (gồm noise sweep ~4–5h) | ~0.5–1 ngày | ngày 3–4 |
| Phase H | pull results + **DELETE VM** | ~15 phút | ngày 3–4 |

→ Xong ~**ngày 3.5–4 → XÓA VM**. Phần viết bài (D4→D13) làm **local, miễn phí**.

---

## 💰 Chi phí & giữ trong $300 free credit

Giá c2d-standard-56 on-demand ~**$3.0/h (region US)** / ~**$3.7/h (Singapore)** — *ước tính, xác
nhận lại số $/h GCP hiển thị khi tạo máy*. **Tiền = $/h × số giờ VM SỐNG** (không phải giờ tính toán).

| Kịch bản | Giờ VM sống | US ~$3.0 | Singapore ~$3.7 |
|---|---|---|---|
| **Đồng thời + xóa ngay (runbook này)** | ~84h | **~$252 ✅** | ~$311 ⚠️ |
| Tuần tự | ~120h | ~$360 ❌ | ~$444 ❌ |
| Quên tắt, để chạy cả tuần viết bài | 168h+ | $500+ ❌ | $620+ ❌ |

**4 quy tắc giữ trong $300:**
1. **VM chỉ sống trong cửa sổ compute (~3.5 ngày)** — viết paper làm local. Mỗi ngày quên tắt ≈ **−$72**.
2. **Chạy đồng thời** cả 3 stage (Phase C+E) — đã set sẵn trong runbook này.
3. **Region `us-central1`** (rẻ hơn Singapore ~23%; batch chạy nền, SSH chậm vài chục ms không sao).
4. **Budget Alert $250** (Billing → Budgets) + **DELETE** (không phải Stop) VM ngay khi pull xong results.

---

## Sự cố thường gặp & cách xử lý

| Triệu chứng | Nguyên nhân | Xử lý |
|---|---|---|
| `import libsumo` lỗi | version SUMO lệch venv | `pip install "libsumo==<ver apt>"` hoặc `pip install eclipse-sumo libsumo` |
| `check_obs_match --sumo` fail | `SUMO_HOME` sai / SUMO chưa cài đúng | `echo $SUMO_HOME` = `/usr/share/sumo`; cài lại sumo-tools |
| 1 job crash giữa chừng | (không có resume mid-train) | **chạy lại cùng `--campaign-id`** → launcher skip job đã có `bench.json`, chỉ chạy lại job dở |
| Rớt SSH, sợ mất job | — | Vô hại nếu chạy trong **tmux**; `tmux attach -t s1` xem lại |
| SPS thấp bất thường | quên pin thread | kiểm `echo $OMP_NUM_THREADS` = 1; nếu trống → `export` rồi chạy lại |
| Tạo VM báo thiếu vCPU | quota account mới | IAM&Admin → Quotas → CPUs → xin tăng |
| `df -h` gần đầy | checkpoint/log nhiều | dọn `results/.../checkpoint_*.pt` trung gian, giữ `best_model.pt` |

## Định nghĩa "DONE" mỗi stage (tick khi đạt)

- [ ] B: `git status` up-to-date với origin/master + sạch, `check_obs_match --sumo` PASS, pilot in PILOT SUMMARY
- [ ] C: `find results/paper1_mappo/main_05M -name bench.json | wc -l` = **30**
- [ ] E: ablation_reward_05M = **18** bench.json, ablation_obs_05M = **9**
- [ ] F: sublane_lanebased có best_model.pt + cross-eval table
- [ ] G: `figures/tables/*.tex` sinh ra; FIG-3/4/5/6 có; Abstract X/Y/Z/W% điền được
- [ ] H: results/figures đã tải về local; **VM đã DELETE**; Budget alert đã đặt

## ⚠️ 4 quy tắc vàng
1. Mọi launcher chạy **trong tmux** (rớt SSH không mất job).
2. **Verify Phase B PASS** trước khi launch (kẻo đốt compute mới phát hiện SUMO lỗi).
3. **Chạy đồng thời** cả 3 stage (30+17+9 worker) — giữ máy 100% bận, ~2.5 ngày, vừa $300.
4. **DELETE VM** ngay khi pull xong results (đừng để sống qua giai đoạn viết bài); Budget Alert $250.
