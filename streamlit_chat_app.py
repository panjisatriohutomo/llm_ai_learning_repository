# Import library yang dibutuhkan
import streamlit as st          # framework web app
from google import genai         # SDK Gemini dari Google

# ── 1. Konfigurasi Halaman ───────────────────────────────────────────────────
# st.title() dan st.caption() menampilkan judul dan keterangan di bagian atas
st.title("Gemini Chatbot")
st.caption("Chatbot sederhana menggunakan Google Gemini Flash")

# ── 2. Sidebar: Pengaturan App ───────────────────────────────────────────────
# Semua widget di dalam blok 'with st.sidebar:' akan muncul di panel samping
with st.sidebar:
    st.subheader("Pengaturan")

    # Kotak input untuk API key
    # type="password" menyembunyikan teks yang diketik (muncul sebagai titik-titik)
    google_api_key = st.text_input("Google AI API Key", type="password")

    # Tombol untuk mereset percakapan
    # Parameter 'help' menampilkan tooltip saat kursor diarahkan ke tombol
    reset_button = st.button("Reset Percakapan", help="Hapus semua pesan dan mulai dari awal")

# ── 3. Validasi API Key ──────────────────────────────────────────────────────
# Kalau user belum memasukkan API key, tampilkan pesan dan hentikan eksekusi
if not google_api_key:
    st.info("Masukkan Google AI API Key di sidebar untuk mulai chat.", icon="🗝️")
    # st.stop() menghentikan eksekusi skrip di titik ini
    # Kode setelah st.stop() tidak akan dijalankan
    st.stop()

# ── 4. Inisialisasi Gemini Client ────────────────────────────────────────────
# Bagian ini hanya membuat client baru kalau:
# - Client belum pernah dibuat (pertama kali app dijalankan), ATAU
# - User mengganti API key di sidebar
#
# Kenapa perlu dicek seperti ini?
# Karena setiap interaksi user menyebabkan seluruh skrip dijalankan ulang.
# Tanpa pengecekan ini, kita akan membuat client baru setiap kali user ketik pesan
# — yang artinya konteks percakapan akan hilang terus.
#
# getattr(obj, 'attr', default) = cara aman mengakses atribut yang mungkin belum ada
if ("genai_client" not in st.session_state) or (
    getattr(st.session_state, "_last_key", None) != google_api_key
):
    try:
        # Buat client Gemini baru dengan API key dari sidebar
        st.session_state.genai_client = genai.Client(api_key=google_api_key)

        # Simpan key yang dipakai — untuk deteksi perubahan key nanti
        st.session_state._last_key = google_api_key

        # Kalau key berganti, hapus session chat lama
        # .pop() menghapus key dari dict dengan aman (tidak error kalau key tidak ada)
        st.session_state.pop("chat", None)
        st.session_state.pop("messages", None)

    except Exception as e:
        st.error(f"API Key tidak valid: {e}")
        st.stop()

# ── 5. Inisialisasi Chat Session & Riwayat Pesan ────────────────────────────
# Inisialisasi chat session Gemini kalau belum ada
if "chat" not in st.session_state:
    # Buat chat session baru dengan model gemini-2.5-flash
    # Session ini menyimpan konteks percakapan di sisi Gemini
    st.session_state.chat = st.session_state.genai_client.chats.create(
        model="gemini-2.5-flash"
    )

# Inisialisasi list riwayat pesan kalau belum ada
# List ini menyimpan semua pesan untuk ditampilkan kembali saat skrip dijalankan ulang
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── 6. Tombol Reset ──────────────────────────────────────────────────────────
# Kalau tombol reset diklik, hapus chat session dan riwayat pesan
if reset_button:
    st.session_state.pop("chat", None)
    st.session_state.pop("messages", None)
    # st.rerun() memaksa Streamlit me-refresh halaman dari awal
    # Ini akan menjalankan ulang seluruh skrip, dan karena session sudah dihapus,
    # chat baru akan dibuat di langkah 5
    st.rerun()

# ── 7. Tampilkan Riwayat Percakapan ─────────────────────────────────────────
# Loop ini menampilkan semua pesan yang sudah ada di session_state
# Setiap kali skrip dijalankan ulang, pesan-pesan ini ditampilkan kembali
for msg in st.session_state.messages:
    # st.chat_message() membuat bubble chat dengan role yang sesuai
    # role "user" = bubble di kanan, role "assistant" = bubble di kiri
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 8. Input & Respons ───────────────────────────────────────────────────────
# st.chat_input() membuat kotak input di bagian bawah halaman
# Nilai yang diketik user tersimpan di variabel 'prompt'
# Variabel ini bernilai None kalau user belum mengirim pesan
prompt = st.chat_input("Ketik pesanmu di sini...")

# Hanya jalankan bagian ini kalau user mengirim pesan
if prompt:
    # Langkah 1: Tambah pesan user ke riwayat
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Langkah 2: Tampilkan bubble pesan user
    with st.chat_message("user"):
        st.markdown(prompt)

    # Langkah 3: Kirim ke Gemini dan tampilkan respons
    try:
        # Kirim pesan ke Gemini melalui chat session yang sudah ada
        # chat.send_message() secara otomatis menyertakan riwayat percakapan sebelumnya
        response = st.session_state.chat.send_message(prompt)

        # Ambil teks dari respons
        # hasattr() memeriksa apakah objek punya atribut tertentu
        # Ini mencegah error kalau format respons tidak terduga
        if hasattr(response, "text"):
            answer = response.text
        else:
            answer = str(response)

    except Exception as e:
        # Kalau ada error (misal: rate limit, koneksi putus), tampilkan pesan error
        answer = f"Terjadi error: {e}"

    # Langkah 4: Tampilkan bubble respons assistant
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Langkah 5: Simpan respons ke riwayat
    st.session_state.messages.append({"role": "assistant", "content": answer})
