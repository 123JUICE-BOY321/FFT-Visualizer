import streamlit as st
import numpy as np
import plotly.graph_objects as go
import cv2
from skimage import data, img_as_float

st.set_page_config(layout="wide", page_title="Filter Design Presentation", initial_sidebar_state="expanded")

@st.cache_data
def load_image(key):
    imgs = {
        "Cameraman": img_as_float(data.camera()),
        "Moon": img_as_float(data.moon()),
        "Text": img_as_float(data.page()),
        "Brain (MRI)": img_as_float(data.brain()[0])
    }
    return imgs.get(key, imgs["Cameraman"])

def distance(shape):
    h, w = shape
    ch, cw = h // 2, w // 2
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    return np.sqrt((X - cw)**2 + (Y - ch)**2)

def make_filter(shape, f_type, D0, order, mode):
    D = distance(shape)
    if f_type == "Ideal":
        H = (D <= D0).astype(float)
    elif f_type == "Butterworth":
        H = 1 / (1 + (D / (D0 + 1e-9))**(2 * order))
    elif f_type == "Gaussian":
        H = np.exp(-(D**2) / (2 * D0**2))
    if mode == "Highpass":
        H = 1 - H
    return H

def apply_filter(img, H):
    img32 = img.astype(np.float32)
    F = cv2.dft(img32, flags=cv2.DFT_COMPLEX_OUTPUT)
    F = np.fft.fftshift(F, axes=(0, 1)) 
    magF = cv2.magnitude(F[:,:,0], F[:,:,1])
    magF = np.log1p(magF)
    magF = (magF - magF.min()) / (magF.max() - magF.min()) if magF.max() > magF.min() else np.zeros_like(magF)
    H2 = cv2.merge([H.astype(np.float32), H.astype(np.float32)])
    G = F * H2
    magG = cv2.magnitude(G[:,:,0], G[:,:,1])
    magG = np.log1p(magG)
    magG = (magG - magG.min()) / (magG.max() - magG.min()) if magG.max() > magG.min() else np.zeros_like(magG)
    G = np.fft.ifftshift(G, axes=(0, 1)) 
    res = cv2.idft(G)
    res = cv2.magnitude(res[:,:,0], res[:,:,1])
    res = cv2.normalize(res, None, 0, 1, cv2.NORM_MINMAX)
    res = np.clip(res, 0.0, 1.0)
    return res, magF, magG

def run_filter_visualization(img_key, f_type, D0, order, mode):
    img = load_image(img_key)
    H = make_filter(img.shape, f_type, D0, order, mode)
    res, magF, magG = apply_filter(img, H)
    left, right = st.columns([1, 1.2])
    with left:
        st.markdown("### Frequency Response Mask")
        fig = go.Figure(go.Surface(
            z=H[::4, ::4],
            colorscale="Viridis",
            showscale=True
        ))
        fig.update_layout(
            height=600,
            margin=dict(l=0, r=0, t=100, b=0),
            scene=dict(
                xaxis_title="Frequency X",
                yaxis_title="Frequency Y",
                zaxis_title="Amplitude",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            )
        )
        fig.update_traces(
            contours_x=dict(show=True, usecolormap=True, highlightcolor="white"),
            contours_y=dict(show=True, usecolormap=True, highlightcolor="white"),
            contours_z=dict(show=True, usecolormap=True, highlightcolor="white")
        )
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("### Processing Stages")
        a, b = st.columns(2)
        c, d = st.columns(2)
        a.image(img, caption="Original Image", width="stretch")
        b.image(magF, caption="Frequency Spectrum", width="stretch")
        c.image(magG, caption="Filtered Spectrum", width="stretch")
        d.image(res, caption="Filtered Image", width="stretch")

    return H

SLIDES = [
    "1. Introduction",
    "2. Ideal Lowpass", "3. Butterworth Lowpass", "4. Gaussian Lowpass",
    "5. Ideal Highpass", "6. Butterworth Highpass", "7. Gaussian Highpass",
    "8. Conclusion"
]

if "slide_idx" not in st.session_state:
    st.session_state.slide_idx = 0

def next_slide():
    if st.session_state.slide_idx < len(SLIDES) - 1:
        st.session_state.slide_idx += 1

def prev_slide():
    if st.session_state.slide_idx > 0:
        st.session_state.slide_idx -= 1

with st.sidebar:
    st.header("Navigation")
    selected = st.radio("Go to:", SLIDES, index=st.session_state.slide_idx)
    if SLIDES.index(selected) != st.session_state.slide_idx:
        st.session_state.slide_idx = SLIDES.index(selected)
        st.rerun()
    st.markdown("---")
    title = SLIDES[st.session_state.slide_idx]
    if "Lowpass" in title or "Highpass" in title or "Introduction" in title:
        st.header("Controls")
        img_key = st.selectbox("Test Image", ["Cameraman", "Moon", "Text", "Brain (MRI)"])
        if "Lowpass" in title or "Highpass" in title:
            D0 = st.slider("Cutoff Frequency (D0)", 5, 100, 30)
            order = 1
            if "Butterworth" in title:
                order = st.slider("Filter Order (n)", 1, 10, 2)

title = SLIDES[st.session_state.slide_idx]

if title == "1. Introduction":
    st.title("Frequency Domain Filtering")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Spatial Domain
        * Image: $f(x,y)$
        * $(x,y)$ → pixel coordinates
        * Processing is performed directly on intensity values
        """)
        st.markdown("""
        ### Frequency Domain
        * Image: $F(u,v)$
        * $(u,v)$ → frequency coordinates
        * Describes rate of intensity variation
        """)
    with col2:
        st.image("Assets/frequency_in_images.png", width="stretch", caption="Frequency in Images")
    st.markdown("---")
    st.markdown("### Convolution Theorem")
    st.markdown("In spatial domain:")
    st.latex(r"g(x,y) = f(x,y) * h(x,y)")
    st.markdown("In frequency domain:")
    st.latex(r"G(u,v) = F(u,v)\,H(u,v)")
    st.markdown("""Convolution in spatial domain becomes multiplication in frequency domain.""")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Fourier Transform")
        st.latex(r"F(u,v)=\int_{-\infty}^{\infty}\int_{-\infty}^{\infty} f(x,y)e^{-j2\pi(ux+vy)}dxdy")
        st.markdown("""Represents continuous signals as a sum of complex exponentials.""")
    with col2:
        st.markdown("### Inverse Fourier Transform")
        st.latex(r"f(x,y)=\int_{-\infty}^{\infty}\int_{-\infty}^{\infty} F(u,v)e^{j2\pi(ux+vy)}dudv")
        st.markdown("""Reconstructs the spatial signal from its frequency components.""")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Discrete Fourier Transform")
        st.latex(r"F(u,v)=\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)e^{-j2\pi\left(\frac{ux}{M}+\frac{vy}{N}\right)}")
    with col2:
        st.markdown("### Discrete Inverse Fourier Transform")
        st.latex(r"f(x,y)=\frac{1}{MN}\sum_{u=0}^{M-1}\sum_{v=0}^{N-1}F(u,v)e^{j2\pi\left(\frac{ux}{M}+\frac{vy}{N}\right)}")
    st.markdown("Images are discrete and finite, so we use the DFT.")
    st.markdown("""The DFT decomposes the image into discrete frequency components, and the IDFT reconstructs the spatial image.""")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Steps for Frequency Domain Filtering")
        st.markdown("""
        1. Start with image $f(x,y)$  
        2. Center by multiplying by $(-1)^{x+y}$
        3. Compute DFT to get $F(u,v)$  
        4. Multiply by a filter function $H(u,v) \\rightarrow G(u,v) = H(u,v)F(u,v)$  
        5. Compute Inverse DFT  
        6. De-center by multiplying again by $(-1)^{x+y}$  
        7. Take the real part to get the final image  
        """)
    with col2:
        img = load_image(img_key)
        img32 = img.astype(np.float32)
        F = cv2.dft(img32, flags=cv2.DFT_COMPLEX_OUTPUT)
        F = np.fft.fftshift(F, axes=(0, 1))
        magF = cv2.magnitude(F[:,:,0], F[:,:,1])
        magF = np.log1p(magF)
        magF = (magF - magF.min()) / (magF.max() - magF.min())
        col3, col4 = st.columns(2)
        with col3:
            st.image(img, width="stretch", caption="Spatial Domain")
        with col4:
            st.image(magF, width="stretch", caption="Frequency Domain")
    with st.expander("Why multiply by $(-1)^{x+y}$ ?"):
        st.markdown("""
        Normally in the DFT result:
        * Low frequencies appear at the top-left corner.
        * High frequencies appear at the bottom-right corner.\n
        But for image processing we want:
        * Low frequencies appear in the center of the spectrum.\n
        Multiplying by $(-1)^{x+y}$ shifts the spectrum so that the **DC component (zero frequency)** moves to the center.
        """)
    # with st.expander("Why use logarithmic transform for the magnitude spectrum?"):
    #     st.markdown("""
    #     Normally in the magnitude spectrum:
    #     * A few frequencies have very large values.
    #     * Most frequencies have very small values.\n
    #     This large range makes the spectrum hard to visualize.\n
    #     Applying
    #     $$
    #     \log(1 + |F(u,v)|)
    #     $$
    #     compresses the range so more frequency details become visible.
    #     """)
    

elif title == "2. Ideal Lowpass":
    st.title("Ideal Lowpass Filter")
    st.markdown("An Ideal Lowpass filter preserves low frequencies and removes high frequencies using a sharp cutoff at $D_0$.")
    st.markdown("---")
    left, right = st.columns(2)
    with left:
        st.markdown("### Frequency Response Function")
        st.latex(r"H(u,v)=\begin{cases}1 & \text{if } D(u,v)\le D_0 \\ 0 & \text{if } D(u,v)>D_0\end{cases}")
        st.markdown("**Where:**")
        st.markdown(r"$H(u,v)$ → Filter transfer function")
        st.markdown(r"$D(u,v)$ → Distance from frequency center")
        st.markdown(r"$D_0$ → Cutoff frequency")
    with right:
        st.markdown("### Code Snippet (OpenCV)")
        st.code(
            "img32 = img.astype(np.float32)\n"
            "F = cv2.dft(img32, flags=cv2.DFT_COMPLEX_OUTPUT)\n"
            "F = np.fft.fftshift(F, axes=(0, 1))\n\n"
            "H = (D <= D0).astype(np.float32)\n"
            "H = cv2.merge([H, H])\n\n"
            "G = F * H\n"
            "G = np.fft.ifftshift(G, axes=(0, 1))\n"
            "res = cv2.idft(G)\n"
            "res = cv2.magnitude(res[:,:,0], res[:,:,1])",
            language="python"
        )
    st.markdown("---")
    H = run_filter_visualization(img_key, "Ideal", D0, order, mode="Lowpass")
    st.markdown("---")
    impulse_response = np.fft.ifftshift(H, axes=(0, 1))
    h_spatial = np.fft.ifft2(impulse_response)
    h_spatial = np.fft.fftshift(h_spatial, axes=(0, 1))
    h_spatial = np.real(h_spatial)
    h_spatial = h_spatial / np.max(np.abs(h_spatial))
    center_row = h_spatial[h_spatial.shape[0] // 2, :]
    x = np.arange(len(center_row)) - len(center_row) // 2
    r1, r2 = st.columns([1, 1.2])
    with r1:
        st.markdown("### Spatial Impulse Response")
        fig_ring = go.Figure(go.Scatter(x=x, y=center_row, line=dict(color="#ff4b4b")))
        fig_ring.update_layout(
            height=300, 
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Distance from Center (pixels)",
            yaxis_title="Amplitude"
        )
        st.plotly_chart(fig_ring, use_container_width=True)
    with r2:
        st.markdown("### The Gibbs Phenomenon (Ringing)")
        st.markdown("""
        * **Cause:** Ideal filters have a sharp frequency cutoff.
        * **Result:** The inverse transform becomes a **sinc function** (left) that extends infinitely in the spatial domain.
        * **Effect:** Ripples appear around edges, producing **ringing artifacts**.
        """)
        st.latex(r"sinc(x) = \frac{\sin(\pi x)}{\pi x}")

elif title == "3. Butterworth Lowpass":
    st.title("Butterworth Lowpass Filter")
    st.markdown("A Butterworth Lowpass filter provides a smooth transition between passband and stopband, controlled by the filter order $n$.")
    st.markdown("---")
    left, right = st.columns(2)
    with left:
        st.markdown("### Frequency Response Function")
        st.latex(r"H(u,v)=\frac{1}{1+\left(\frac{D(u,v)}{D_0}\right)^{2n}}")
        st.markdown("**Where:**")
        st.markdown(r"$H(u,v)$ → Filter transfer function")
        st.markdown(r"$D(u,v)$ → Distance from frequency center")
        st.markdown(r"$D_0$ → Cutoff frequency")
        st.markdown(r"$n$ → Filter order")
    with right:
        st.markdown("### Code Snippet (OpenCV)")
        st.code(
            "img32 = img.astype(np.float32)\n"
            "F = cv2.dft(img32, flags=cv2.DFT_COMPLEX_OUTPUT)\n"
            "F = np.fft.fftshift(F, axes=(0, 1))\n\n"
            "H = 1 / (1 + (D / D0)**(2 * order))\n"
            "H = cv2.merge([H.astype(np.float32), H.astype(np.float32)])\n\n"
            "G = F * H\n"
            "G = np.fft.ifftshift(G, axes=(0, 1))\n"
            "res = cv2.idft(G)\n"
            "res = cv2.magnitude(res[:,:,0], res[:,:,1])",
            language="python"
        )
    st.markdown("---")
    run_filter_visualization(img_key, "Butterworth", D0, order, mode="Lowpass")
    st.markdown("---")

elif title == "4. Gaussian Lowpass":
    st.title("Gaussian Lowpass Filter")
    st.markdown("A Gaussian Lowpass filter attenuates high frequencies smoothly using a Gaussian function, producing no sharp cutoff.")
    st.markdown("---")
    left, right = st.columns(2)
    with left:
        st.markdown("### Frequency Response Function")
        st.latex(r"H(u,v)=e^{-\frac{D(u,v)^2}{2D_0^2}}")
        st.markdown("**Where:**")
        st.markdown(r"$H(u,v)$ → Filter transfer function")
        st.markdown(r"$D(u,v)$ → Distance from frequency center")
        st.markdown(r"$D_0$ → Cutoff frequency")
    with right:
        st.markdown("### Code Snippet (OpenCV)")
        st.code(
            "img32 = img.astype(np.float32)\n"
            "F = cv2.dft(img32, flags=cv2.DFT_COMPLEX_OUTPUT)\n"
            "F = np.fft.fftshift(F, axes=(0, 1))\n\n"
            "H = np.exp(-(D**2) / (2 * D0**2))\n"
            "H = cv2.merge([H.astype(np.float32), H.astype(np.float32)])\n\n"
            "G = F * H\n"
            "G = np.fft.ifftshift(G, axes=(0, 1))\n"
            "res = cv2.idft(G)\n"
            "res = cv2.magnitude(res[:,:,0], res[:,:,1])",
            language="python"
        )
    st.markdown("---")
    run_filter_visualization(img_key, "Gaussian", D0, order, mode="Lowpass")
    st.markdown("---")

elif title == "5. Ideal Highpass":
    st.title("Ideal Highpass Filter")
    st.markdown("An Ideal Highpass filter suppresses low frequencies and preserves high frequencies using a sharp cutoff at $D_0$.")
    st.markdown("---")
    left, right = st.columns(2)
    with left:
        st.markdown("### Frequency Response Function")
        st.latex(r"H(u,v)=\begin{cases}0 & \text{if } D(u,v)\le D_0 \\ 1 & \text{if } D(u,v)>D_0\end{cases}")
        st.markdown("**Where:**")
        st.markdown(r"$H(u,v)$ → Filter transfer function")
        st.markdown(r"$D(u,v)$ → Distance from frequency center")
        st.markdown(r"$D_0$ → Cutoff frequency")
    with right:
        st.markdown("### Code Snippet (OpenCV)")
        st.code(
            "img32 = img.astype(np.float32)\n"
            "F = cv2.dft(img32, flags=cv2.DFT_COMPLEX_OUTPUT)\n"
            "F = np.fft.fftshift(F, axes=(0, 1))\n\n"
            "H = (D > D0).astype(np.float32)\n"
            "H = cv2.merge([H, H])\n\n"
            "G = F * H\n"
            "G = np.fft.ifftshift(G, axes=(0, 1))\n"
            "res = cv2.idft(G)\n"
            "res = cv2.magnitude(res[:,:,0], res[:,:,1])",
            language="python"
        )
    st.markdown("---")
    run_filter_visualization(img_key, "Ideal", D0, order, mode="Highpass")
    st.markdown("---")

elif title == "6. Butterworth Highpass":
    st.title("Butterworth Highpass Filter")
    st.markdown("A Butterworth Highpass filter smoothly attenuates low frequencies while preserving high frequencies, controlled by the filter order $n$.")
    st.markdown("---")
    left, right = st.columns(2)
    with left:
        st.markdown("### Frequency Response Function")
        st.latex(r"H(u,v)=\frac{1}{1+\left(\frac{D_0}{D(u,v)}\right)^{2n}}")
        st.markdown("**Where:**")
        st.markdown(r"$H(u,v)$ → Filter transfer function")
        st.markdown(r"$D(u,v)$ → Distance from frequency center")
        st.markdown(r"$D_0$ → Cutoff frequency")
        st.markdown(r"$n$ → Filter order")
    with right:
        st.markdown("### Code Snippet (OpenCV)")
        st.code(
            "img32 = img.astype(np.float32)\n"
            "F = cv2.dft(img32, flags=cv2.DFT_COMPLEX_OUTPUT)\n"
            "F = np.fft.fftshift(F, axes=(0, 1))\n\n"
            "H = 1 / (1 + (D0 / (D + 1e-9))**(2 * order))\n"
            "H = cv2.merge([H.astype(np.float32), H.astype(np.float32)])\n\n"
            "G = F * H\n"
            "G = np.fft.ifftshift(G, axes=(0, 1))\n"
            "res = cv2.idft(G)\n"
            "res = cv2.magnitude(res[:,:,0], res[:,:,1])",
            language="python"
        )
    st.markdown("---")
    run_filter_visualization(img_key, "Butterworth", D0, order, mode="Highpass")
    st.markdown("---")
        
elif title == "7. Gaussian Highpass":
    st.title("Gaussian Highpass Filter")
    st.markdown("A Gaussian Highpass filter smoothly suppresses low frequencies while preserving high frequencies without a sharp cutoff.")
    st.markdown("---")
    left, right = st.columns(2)
    with left:
        st.markdown("### Frequency Response Function")
        st.latex(r"H(u,v)=1 - e^{-\frac{D(u,v)^2}{2D_0^2}}")
        st.markdown("**Where:**")
        st.markdown(r"$H(u,v)$ → Filter transfer function")
        st.markdown(r"$D(u,v)$ → Distance from frequency center")
        st.markdown(r"$D_0$ → Cutoff frequency")
    with right:
        st.markdown("### Code Snippet (OpenCV)")
        st.code(
            "img32 = img.astype(np.float32)\n"
            "F = cv2.dft(img32, flags=cv2.DFT_COMPLEX_OUTPUT)\n"
            "F = np.fft.fftshift(F, axes=(0, 1))\n\n"
            "H = 1 - np.exp(-(D**2) / (2 * D0**2))\n"
            "H = cv2.merge([H.astype(np.float32), H.astype(np.float32)])\n\n"
            "G = F * H\n"
            "G = np.fft.ifftshift(G, axes=(0, 1))\n"
            "res = cv2.idft(G)\n"
            "res = cv2.magnitude(res[:,:,0], res[:,:,1])",
            language="python"
        )
    st.markdown("---")
    run_filter_visualization(img_key, "Gaussian", D0, order, mode="Highpass")
    st.markdown("---")

elif title == "8. Conclusion":
    st.title("Summary & Comparative Analysis")

    st.markdown("""
Frequency domain filtering modifies an image by shaping its frequency spectrum using a transfer function $H(u,v)$.
Each filter differs in how sharply it transitions between passband and stopband, which directly affects spatial behavior.
""")

    st.markdown("---")
    st.subheader("Frequency Response Comparison")

    D0 = st.slider("Shared Cutoff Frequency (D0)", 10, 60, 30)
    order = st.slider("Butterworth Order (n)", 1, 6, 2)

    D = np.linspace(0, 100, 400)

    H_ideal = (D <= D0).astype(float)
    H_bw = 1 / (1 + (D / D0)**(2 * order))
    H_gauss = np.exp(-(D**2) / (2 * D0**2))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=D, y=H_ideal, name="Ideal", line=dict(width=3, dash="dash")))
    fig.add_trace(go.Scatter(x=D, y=H_bw, name=f"Butterworth (n={order})", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=D, y=H_gauss, name="Gaussian", line=dict(width=3)))

    fig.update_layout(
        xaxis_title="Frequency Distance (D)",
        yaxis_title="H(u,v)",
        height=420
    )

    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("Behavioral Characteristics")

    st.markdown("""
| Filter Type | Transition Sharpness | Ringing Effect | Typical Use Cases |
|:------------|:--------------------|:--------------|:------------------|
| **Ideal** | Abrupt (discontinuous) | High (Gibbs effect) | Theoretical demonstrations |
| **Butterworth** | Controlled by order $n$ | Moderate to Low | Engineering applications |
| **Gaussian** | Smooth and continuous | Minimal / None | Natural & medical images |
""")

    st.markdown("---")
    st.subheader("Key Insight")

    st.markdown("""
- Sharper transitions in frequency domain produce stronger oscillations in spatial domain.
- Ideal filters introduce ringing due to abrupt cutoff.
- Butterworth filters provide a controllable compromise.
- Gaussian filters ensure smooth behavior with minimal artifacts.

**There is no universally best filter — the choice depends on the application and tolerance to artifacts.**
""")

left, center, right = st.columns([1, 7, 1])
if st.session_state.slide_idx > 0:
    left.button("⬅️ Previous", on_click=prev_slide)
if st.session_state.slide_idx < len(SLIDES) - 1:
    right.button("Next ➡️", on_click=next_slide)