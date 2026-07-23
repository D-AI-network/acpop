# PopField AI Smart Cooling — Streamlit Mobile Demo

## Files
- `streamlit_app.py`: mobile-friendly UI
- `demo_v2_backend.py`: PopField/HVAC backend from the validated demo v2 code
- `requirements.txt`: Python dependencies

## Required model assets
Put these beside `streamlit_app.py` before launching:
- `best.pt`
- `Case Info 200 DesignPoints - 최종본.xlsx`

The app also supports uploading these two files from the **모델 연결** expander.
The original `Field data.zip` is not required for demo inference because the checkpoint contains model weights, XYZ coordinates, and normalization statistics; Case Info is used to recover the action/load levels.

## Run locally / Colab
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Colab tunnel example
Streamlit itself runs inside Colab, so a public URL needs a tunnel service. For a hackathon deployment, GitHub + Streamlit Community Cloud is simpler.

## Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload `streamlit_app.py`, `demo_v2_backend.py`, `requirements.txt`, `best.pt`, and the Case Info xlsx.
3. In Streamlit Community Cloud, select `streamlit_app.py` as the entry point.
4. Open the generated URL on a phone or convert it to a QR code for judging/demo.

> Note: GitHub rejects single files above 100 MB. If `best.pt` exceeds that limit, store it externally/private storage and adapt the app to download it at startup, or use another deployment service.

## User flow
1. Select target temperature (22–28°C).
2. Select external / meeting / server / working load levels.
3. Select Balanced / Comfort / Eco.
4. Tap **AI 최적 냉방 찾기**.
5. The backend evaluates 54 HVAC candidates and displays:
   - feasible / near-feasible / infeasible
   - L/M/R direction
   - CMM
   - supply temperature
   - mean temperature / P95
   - zone spread / hotspot / coldspot
   - predicted spatial temperature map

## Interpretation limits
- Current CFD data are steady-state, not dynamic closed-loop trajectories.
- Sensible cooling capacity is thermal capacity, not measured electric power or electricity cost.
- Official Zone masks are not included in the supplied field data; the backend uses four XY quadrants as placeholders unless an official zone JSON is provided.
