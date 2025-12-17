import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.integrate import solve_ivp

st.set_page_config(page_title="TA-11 Robertson Stiff ODE", layout="wide")

# ------------------------
# Model: Robertson (stiff)
# ------------------------
def robertson(t, y):
    y1, y2, y3 = y
    dy1 = -0.04 * y1 + 1e4 * y2 * y3
    dy2 = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * (y2**2)
    dy3 = 3e7 * (y2**2)
    return np.array([dy1, dy2, dy3])

def euler_explicit(f, t0, tf, y0, h, fail_threshold=1e6):
    n = int(np.ceil((tf - t0) / h))
    t = np.zeros(n + 1)
    y = np.zeros((n + 1, len(y0)))
    t[0] = t0
    y[0] = y0

    status = "OK"
    for i in range(n):
        t[i + 1] = t[i] + h
        y[i + 1] = y[i] + h * f(t[i], y[i])

        if not np.isfinite(y[i + 1]).all():
            status = "GAGAL (NaN/Inf)"
            t, y = t[: i + 2], y[: i + 2]
            break
        if np.max(np.abs(y[i + 1])) > fail_threshold:
            status = f"GAGAL (meledak > {fail_threshold:g})"
            t, y = t[: i + 2], y[: i + 2]
            break
        if np.min(y[i + 1]) < -1e-8:
            status = "GAGAL (nilai negatif / tidak fisik)"
            t, y = t[: i + 2], y[: i + 2]
            break

    return t, y, status

def make_t_eval(t0, tf, mode="log"):
    if mode == "log":
        # gabungan linear awal + logspace panjang
        a = np.linspace(t0, min(tf, 1.0), 400)
        if tf > 1.0:
            b = np.logspace(0, np.log10(tf), 800)
            b = b[(b >= t0) & (b <= tf)]
            t_eval = np.unique(np.concatenate([a, b]))
        else:
            t_eval = a
        return t_eval
    else:
        return np.linspace(t0, tf, 1500)

def solve(method, t0, tf, y0, t_eval):
    start = time.time()
    sol = solve_ivp(robertson, (t0, tf), y0, method=method, t_eval=t_eval)
    elapsed = time.time() - start
    return sol, elapsed

# ------------------------
# UI
# ------------------------
st.title("TA-11 — Simulasi Sistem Kaku Robertson (Streamlit)")
st.write("Bandingkan Euler (eksplisit) vs RK45 vs solver stiff (BDF/Radau/LSODA) untuk sistem Robertson.")

with st.sidebar:
    st.header("Parameter")
    t0 = st.number_input("t0", value=0.0, format="%.6f")
    tf = st.number_input("tf", value=1e5, format="%.6f")
    h = st.number_input("h Euler", value=0.01, format="%.6f", min_value=1e-8)
    y1_0 = st.number_input("y1(0)", value=1.0, format="%.6f")
    y2_0 = st.number_input("y2(0)", value=0.0, format="%.6f")
    y3_0 = st.number_input("y3(0)", value=0.0, format="%.6f")
    t_eval_mode = st.selectbox("Mode t_eval", ["log", "linear"])
    run_lsoda = st.checkbox("Jalankan LSODA (jika tersedia)", value=True)

    st.header("Metode solve_ivp")
    methods = st.multiselect(
        "Pilih metode",
        ["RK45", "Radau", "BDF", "LSODA"],
        default=["RK45", "Radau", "BDF"]
    )
    if run_lsoda and "LSODA" not in methods:
        methods.append("LSODA")

run = st.button("Run Simulasi")

# ------------------------
# Run
# ------------------------
if run:
    y0 = np.array([y1_0, y2_0, y3_0], dtype=float)
    t_eval = make_t_eval(t0, tf, mode=t_eval_mode)

    # Euler
    start = time.time()
    t_eu, y_eu, status_eu = euler_explicit(robertson, t0, tf, y0, h)
    time_eu = time.time() - start

    # solve_ivp methods
    results = []
    sols = {}

    # Euler row
    results.append({
        "Metode": f"Euler eksplisit (h={h})",
        "Waktu (detik)": time_eu,
        "nfev/steps": len(t_eu) - 1,
        "Status": status_eu
    })

    for m in methods:
        try:
            sol, elapsed = solve(m, t0, tf, y0, t_eval)
            sols[m] = sol
            results.append({
                "Metode": m,
                "Waktu (detik)": elapsed,
                "nfev/steps": sol.nfev,
                "Status": sol.message
            })
        except Exception as e:
            results.append({
                "Metode": m,
                "Waktu (detik)": None,
                "nfev/steps": None,
                "Status": f"ERROR: {e}"
            })

    df = pd.DataFrame(results)

    # ------------------------
    # Output layout
    # ------------------------
    col1, col2 = st.columns([1.1, 1.0])

    with col1:
        st.subheader("Tabel Kinerja")
        st.dataframe(df, use_container_width=True)

        st.subheader("Euler (bukti gagal / tidak stabil)")
        fig = plt.figure(figsize=(9, 4))
        plt.plot(t_eu, y_eu[:, 0], label="y1")
        plt.plot(t_eu, y_eu[:, 1], label="y2")
        plt.plot(t_eu, y_eu[:, 2], label="y3")
        if tf > 0 and t_eval_mode == "log":
            plt.xscale("log")
        plt.xlabel("t")
        plt.ylabel("konsentrasi")
        plt.title(f"Euler eksplisit — {status_eu}")
        plt.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("solve_ivp Results")
        if len(sols) == 0:
            st.info("Tidak ada solver yang berhasil dijalankan.")
        else:
            chosen = st.selectbox("Pilih hasil untuk ditampilkan", list(sols.keys()))
            sol = sols[chosen]

            fig2 = plt.figure(figsize=(9, 4))
            plt.plot(sol.t, sol.y[0], label="y1")
            plt.plot(sol.t, sol.y[1], label="y2")
            plt.plot(sol.t, sol.y[2], label="y3")
            if tf > 0 and t_eval_mode == "log":
                plt.xscale("log")
            plt.xlabel("t")
            plt.ylabel("konsentrasi")
            plt.title(f"Robertson — {chosen}")
            plt.legend()
            st.pyplot(fig2)

    st.markdown("---")
    st.subheader("Catatan Analisis (untuk laporan)")
    st.write(
        "- Robertson bersifat stiff karena ada konstanta reaksi sangat besar (mis. 3e7) sehingga ada skala waktu cepat & lambat.\n"
        "- Metode eksplisit cenderung tidak stabil kecuali memakai h sangat kecil.\n"
        "- Solver stiff (BDF/Radau/LSODA) lebih stabil dan biasanya lebih efisien (nfev lebih masuk akal)."
    )
else:
    st.info("Atur parameter di sidebar lalu klik **Run Simulasi**.")
