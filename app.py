import simpy
import random
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# -------------------------
# SimPy model (same logic as notebook)
# -------------------------
def sample_interarrival(mean):
    return random.expovariate(1.0 / mean)

def sample_service_time(tmin, tmax):
    return random.uniform(tmin, tmax)

def car_process(env, car_id, cashier, records, service_min, service_max):
    arrival_time = env.now

    with cashier.request() as req:
        yield req
        start_service = env.now
        waiting_time = start_service - arrival_time

        service_time = sample_service_time(service_min, service_max)
        yield env.timeout(service_time)

        finish_time = env.now

    records.append({
        "car_id": car_id,
        "arrival_time": arrival_time,
        "start_service": start_service,
        "finish_time": finish_time,
        "waiting_time": waiting_time,
        "service_time": service_time,
        "system_time": finish_time - arrival_time
    })

def car_generator(env, cashier, records, mean_interarrival, service_min, service_max):
    car_id = 0
    while True:
        car_id += 1
        env.process(car_process(env, car_id, cashier, records, service_min, service_max))
        yield env.timeout(sample_interarrival(mean_interarrival))

def monitor_queue(env, cashier, queue_log, interval=1.0):
    while True:
        queue_log.append({
            "time": env.now,
            "queue_length": len(cashier.queue),
            "in_service": cashier.count
        })
        yield env.timeout(interval)

def run_simulation(num_cashiers, sim_time, seed, mean_interarrival, service_min, service_max, monitor_interval=1.0):
    random.seed(seed)

    env = simpy.Environment()
    cashier = simpy.Resource(env, capacity=num_cashiers)

    records = []
    queue_log = []

    env.process(car_generator(env, cashier, records, mean_interarrival, service_min, service_max))
    env.process(monitor_queue(env, cashier, queue_log, interval=monitor_interval))

    env.run(until=sim_time)

    df = pd.DataFrame(records)
    qdf = pd.DataFrame(queue_log)

    summary = {
        "num_cashiers": num_cashiers,
        "cars_processed": len(df),
        "avg_wait": float(df["waiting_time"].mean()) if len(df) else 0.0,
        "max_wait": float(df["waiting_time"].max()) if len(df) else 0.0,
        "avg_system": float(df["system_time"].mean()) if len(df) else 0.0,
        "avg_queue_length": float(qdf["queue_length"].mean()) if len(qdf) else 0.0,
        "max_queue_length": float(qdf["queue_length"].max()) if len(qdf) else 0.0,
    }
    return df, qdf, summary

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="TP-14 Drive-Thru SimPy", layout="wide")

st.title("TP-14 — Simulasi Antrean Drive-Thru (SimPy + Streamlit)")
st.write("Aplikasi ini melakukan simulasi antrean mobil drive-thru dengan skenario **1 kasir vs 2 kasir**.")

with st.sidebar:
    st.header("Parameter Simulasi")
    seed = st.number_input("Random Seed", value=42, step=1)
    sim_time = st.number_input("Durasi Simulasi (menit)", value=600, step=50, min_value=50)
    mean_interarrival = st.number_input("Mean Inter-arrival (menit)", value=10.0, step=1.0, min_value=0.1)

    st.subheader("Waktu Pelayanan (Uniform)")
    service_min = st.number_input("Min (menit)", value=3.0, step=0.5, min_value=0.1)
    service_max = st.number_input("Max (menit)", value=5.0, step=0.5, min_value=0.1)

    monitor_interval = st.number_input("Interval Monitor Antrean (menit)", value=1.0, step=0.5, min_value=0.1)

    run_btn = st.button("Run Simulation")

if run_btn:
    if service_max < service_min:
        st.error("Service Max harus >= Service Min.")
        st.stop()

    col1, col2 = st.columns(2)

    # Run Scenario A and B
    df_A, q_A, summary_A = run_simulation(
        num_cashiers=1,
        sim_time=sim_time,
        seed=seed,
        mean_interarrival=mean_interarrival,
        service_min=service_min,
        service_max=service_max,
        monitor_interval=monitor_interval
    )
    df_B, q_B, summary_B = run_simulation(
        num_cashiers=2,
        sim_time=sim_time,
        seed=seed,
        mean_interarrival=mean_interarrival,
        service_min=service_min,
        service_max=service_max,
        monitor_interval=monitor_interval
    )

    summary_table = pd.DataFrame([summary_A, summary_B])

    st.subheader("Ringkasan Hasil (summary_table)")
    st.dataframe(summary_table, use_container_width=True)

    # Improvement calc
    if summary_A["avg_wait"] > 0:
        improvement = (summary_A["avg_wait"] - summary_B["avg_wait"]) / summary_A["avg_wait"] * 100.0
    else:
        improvement = 0.0
    st.success(f"Penurunan rata-rata waktu tunggu (1 kasir -> 2 kasir): {improvement:.2f}%")

    st.divider()
    st.subheader("Visualisasi")

    # Plot 1: waiting time vs car_id
    fig1 = plt.figure(figsize=(10, 4))
    plt.plot(df_A["car_id"], df_A["waiting_time"], label="1 Kasir")
    plt.plot(df_B["car_id"], df_B["waiting_time"], label="2 Kasir")
    plt.xlabel("Urutan Mobil (car_id)")
    plt.ylabel("Waktu Tunggu (menit)")
    plt.title("Line Chart — Waktu Tunggu Mobil")
    plt.legend()
    st.pyplot(fig1, clear_figure=True)

    # Plot 2: histogram system time
    fig2 = plt.figure(figsize=(10, 4))
    plt.hist(df_A["system_time"], bins=20, alpha=0.6, label="1 Kasir")
    plt.hist(df_B["system_time"], bins=20, alpha=0.6, label="2 Kasir")
    plt.xlabel("Total Waktu dalam Sistem (menit)")
    plt.ylabel("Frekuensi")
    plt.title("Histogram — Total Waktu dalam Sistem")
    plt.legend()
    st.pyplot(fig2, clear_figure=True)

    # Plot 3: queue length over time
    fig3 = plt.figure(figsize=(10, 4))
    plt.plot(q_A["time"], q_A["queue_length"], label="1 Kasir")
    plt.plot(q_B["time"], q_B["queue_length"], label="2 Kasir")
    plt.xlabel("Waktu Simulasi (menit)")
    plt.ylabel("Panjang Antrean")
    plt.title("Line Chart — Panjang Antrean terhadap Waktu")
    plt.legend()
    st.pyplot(fig3, clear_figure=True)

    st.divider()
    with col1:
        st.subheader("Contoh Data Log Mobil (Skenario 1 Kasir)")
        st.dataframe(df_A.head(10), use_container_width=True)
    with col2:
        st.subheader("Contoh Data Log Mobil (Skenario 2 Kasir)")
        st.dataframe(df_B.head(10), use_container_width=True)
else:
    st.info("Atur parameter di sidebar, lalu klik **Run Simulation**.")