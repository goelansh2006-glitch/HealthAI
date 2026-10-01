import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path
import base64
import textwrap
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

def render_markdown(content, **kwargs):
    if isinstance(content, str):
        content = textwrap.dedent(content).strip()

    if kwargs.get("unsafe_allow_html", False) and "<" in content and ">" in content:
        kwargs.pop("unsafe_allow_html", None)
        return st.html(content)

    return st.markdown(content, **kwargs)

# PAGE CONFIG

st.set_page_config(
    page_title="HealthAI - Your Health, Powered by Artificial Intelligence.",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# PATHS + MODEL

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "heart_disease_model1.pkl"

BACKGROUND_PATH = BASE_DIR / "img" / "heart_background.png"

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error("The trained model could not be loaded.")
    st.code(str(e))
    st.info(
        "Put heart_disease_model1.pkl in the same folder as app.py "
        "and make sure it was trained with the same 11 original features."
    )
    st.stop()


# HELPER: BACKGROUND IMAGE


def get_background_base64(image_path):
    """Convert the background image to base64 for reliable browser rendering."""
    if not image_path.exists():
        return None

    try:
        image_bytes = image_path.read_bytes()
        encoded = base64.b64encode(image_bytes).decode()

        suffix = image_path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }
        mime = mime_types.get(suffix, "image/png")

        return f"data:{mime};base64,{encoded}"
    except Exception:
        return None

background_image = get_background_base64(BACKGROUND_PATH)

# PDF REPORT GENERATOR

def create_pdf_report(input_data, prediction, risk_percent):
    pdf_buffer = BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=28,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#991b1b"),
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=20,
    )

    section_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#172033"),
        spaceBefore=10,
        spaceAfter=10,
    )

    normal_style = ParagraphStyle(
        "NormalText",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#475569"),
    )

    result_style = ParagraphStyle(
        "Result",
        parent=styles["Heading2"],
        fontSize=17,
        leading=22,
        alignment=TA_CENTER,
        textColor=(
            colors.HexColor("#991b1b")
            if prediction == 1
            else colors.HexColor("#166534")
        ),
        spaceAfter=8,
    )

    story = []

    story.append(Paragraph("HealthAI", title_style))
    story.append(Paragraph("Heart Health Assessment Report", subtitle_style))

    assessment = (
        "Higher Estimated Risk Class"
        if prediction == 1
        else "Lower Estimated Risk Class"
    )

    story.append(Paragraph("Assessment Result", section_style))

    result_data = [
        [Paragraph(f"<b>{assessment}</b>", result_style)],
        [
            Paragraph(
                "Estimated Positive-Class Probability: "
                f"<b>{risk_percent:.1f}%</b>",
                normal_style,
            )
        ],
    ]

    result_table = Table(result_data, colWidths=[165 * mm])
    result_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#fff7f8")
                    if prediction == 1
                    else colors.HexColor("#f0fdf4"),
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    1,
                    colors.HexColor("#fecdd3")
                    if prediction == 1
                    else colors.HexColor("#bbf7d0"),
                ),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ]
        )
    )

    story.append(result_table)
    story.append(Spacer(1, 15))

    story.append(Paragraph("Assessment Details", section_style))

    readable_values = {
        "Age": input_data["Age"].iloc[0],
        "Sex": "Male" if input_data["Sex"].iloc[0] == "M" else "Female",
        "Chest Pain Type": {
            "ATA": "Atypical Angina",
            "NAP": "Non-Anginal Pain",
            "ASY": "Asymptomatic",
            "TA": "Typical Angina",
        }[input_data["ChestPainType"].iloc[0]],
        "Resting Blood Pressure": f"{input_data['RestingBP'].iloc[0]} mmHg",
        "Cholesterol": input_data["Cholesterol"].iloc[0],
        "Fasting Blood Sugar": (
            "Normal" if input_data["FastingBS"].iloc[0] == 0 else "High"
        ),
        "Resting ECG": input_data["RestingECG"].iloc[0],
        "Maximum Heart Rate": f"{input_data['MaxHR'].iloc[0]} bpm",
        "Exercise-related Chest Discomfort": (
            "No"
            if input_data["ExerciseAngina"].iloc[0] == "N"
            else "Yes"
        ),
        "ST Depression (Oldpeak)": input_data["Oldpeak"].iloc[0],
        "ST Segment Slope": input_data["ST_Slope"].iloc[0],
    }

    table_data = [
        [
            Paragraph("<b>Information</b>", normal_style),
            Paragraph("<b>Value</b>", normal_style),
        ]
    ]

    for key, value in readable_values.items():
        table_data.append(
            [
                Paragraph(str(key), normal_style),
                Paragraph(str(value), normal_style),
            ]
        )

    details_table = Table(
        table_data,
        colWidths=[80 * mm, 85 * mm],
        repeatRows=1,
    )

    details_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#f1f5f9"),
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#cbd5e1"),
                ),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.append(details_table)
    story.append(Spacer(1, 18))

    story.append(Paragraph("Disclaimer", section_style))

    disclaimer = (
        "This assessment provides an estimated result based on the "
        "information entered. It is not a medical diagnosis and should "
        "not replace advice, examination or treatment from a qualified "
        "healthcare professional. If you have concerning or emergency "
        "symptoms, seek appropriate medical care immediately."
    )

    story.append(Paragraph(disclaimer, normal_style))
    story.append(Spacer(1, 18))

    story.append(
        Paragraph(
            "HealthAI - Your Health, Powered by Artificial Intelligence.",
            subtitle_style,
        )
    )

    doc.build(story)
    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()

# GLOBAL CSS

background_css = ""

if background_image:
    background_css = f"""
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        background-image:
            linear-gradient(
                rgba(246, 248, 252, 0.91),
                rgba(246, 248, 252, 0.91)
            ),
            url("{background_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        opacity: 0.95;
        z-index: 0;
        pointer-events: none;
    }}
    """
render_markdown(
    f"""
    <style>

       GLOBAL

    .stApp {{
        background: #f6f8fc !important;
        color: #172033;
        position: relative;
    }}

    [data-testid="stAppViewContainer"] {{
        position: relative;
        z-index: 1;
        background: transparent !important;
    }}

    {background_css}

    [data-testid="stAppViewContainer"] .main .block-container {{
        max-width: 1450px !important;
        padding-top: 4rem !important;
        padding-bottom: 3rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
    }}

    /* Hide Streamlit chrome */
#MainMenu {{
    visibility: hidden;
}}

footer {{
    visibility: hidden;
}}

/* Keep header available so sidebar toggle remains visible */
header {{
    visibility: visible;
    background: transparent;
}}

       SIDEBAR

    /* =========================
       HEALTHAI SIDEBAR - FIXED
       ========================= */

    [data-testid="stSidebar"] {{
        isolation: isolate !important;
        position: relative !important;
        z-index: 100 !important;
        min-width: 315px !important;
        max-width: 315px !important;
        background: #111827 !important;
        border-right: 1px solid rgba(255,255,255,0.08) !important;
    }}

    /* Force the complete sidebar surface to use the dark gradient.
       Streamlit has multiple nested containers, so all of them are styled. */
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] > div > div,
    [data-testid="stSidebarContent"],
    [data-testid="stSidebarUserContent"] {{
        background: linear-gradient(
            180deg,
            #172554 0%,
            #16213f 42%,
            #111827 100%
        ) !important;
    }}

    [data-testid="stSidebar"] > div:first-child {{
        width: 315px !important;
        min-height: 100vh !important;
        padding: 1rem 1.05rem !important;
    }}

    /* Keep every sidebar text element readable */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] *,
    [data-testid="stSidebarContent"] * {{
        color: #f8fafc !important;
        opacity: 1 !important;
    }}

    [data-testid="stSidebar"] .brand-title {{
        color: #ffffff !important;
    }}

    [data-testid="stSidebar"] .brand-subtitle {{
        color: #cbd5e1 !important;
    }}

    [data-testid="stSidebar"] .sidebar-section-title {{
        color: #94a3b8 !important;
    }}

    /* Radio navigation */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {{
        background: transparent !important;
        color: #e2e8f0 !important;
        border-radius: 13px !important;
        padding: 11px 12px !important;
        min-height: 48px !important;
        border: 1px solid transparent !important;
    }}

    [data-testid="stSidebar"] [data-testid="stRadio"] label p {{
        color: #e2e8f0 !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }}

    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
        background: rgba(255,255,255,0.08) !important;
        border-color: rgba(255,255,255,0.08) !important;
    }}

    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{
        background: linear-gradient(
            90deg,
            rgba(244,63,94,0.30),
            rgba(236,72,153,0.16)
        ) !important;
        border: 1px solid rgba(251,113,133,0.35) !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.12) !important;
    }}

    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {{
        color: #ffffff !important;
        font-weight: 700 !important;
    }}

    [data-testid="stSidebar"] [data-testid="stRadio"] label > div {{
        color: #e2e8f0 !important;
    }}

    .sidebar-brand {{
        text-align: center;
        padding: 22px 8px 24px 8px;
    }}

    .brand-heart {{
        font-size: 58px;
        line-height: 1;
        margin-bottom: 15px;
        filter: drop-shadow(0 7px 12px rgba(236,72,153,0.28));
    }}

    .brand-title {{
        font-size: 24px;
        font-weight: 800;
        letter-spacing: 0.2px;
        color: #ffffff;
    }}

    .brand-subtitle {{
        font-size: 13px;
        margin-top: 7px;
        color: #cbd5e1 !important;
        line-height: 1.5;
    }}

    .sidebar-divider {{
        height: 1px;
        background: rgba(255,255,255,0.10);
        margin: 3px 0 24px 0;
    }}

    .sidebar-section-title {{
        color: #94a3b8 !important;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        padding: 0 8px 10px 8px;
    }}

    /* Radio navigation */
    section[data-testid="stSidebar"] [data-testid="stRadio"] > div {{
        gap: 8px;
    }}

    section[data-testid="stSidebar"] [data-testid="stRadio"] label {{
        background: transparent;
        border-radius: 13px;
        padding: 11px 12px !important;
        min-height: 48px;
        transition: all 0.2s ease;
    }}

    section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
        background: rgba(255,255,255,0.07);
    }}

    section[data-testid="stSidebar"] [data-testid="stRadio"] label p {{
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #e2e8f0 !important;
    }}

    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{
        background: linear-gradient(
            90deg,
            rgba(244,63,94,0.18),
            rgba(236,72,153,0.10)
        );
        border: 1px solid rgba(251,113,133,0.22);
    }}

    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {{
        color: #ffffff !important;
        font-weight: 700 !important;
    }}

    .sidebar-bottom {{
        text-align: center;
        color: #94a3b8 !important;
        font-size: 11px;
        line-height: 1.6;
        padding: 20px 10px 5px;
    }}

    .sidebar-bottom-heart {{
        font-size: 18px;
        margin-bottom: 5px;
    }}

       HERO

    .hero {{
        background:
            linear-gradient(
                135deg,
                rgba(255,241,242,0.96) 0%,
                rgba(255,255,255,0.95) 55%,
                rgba(239,246,255,0.96) 100%
            );
        border: 1px solid rgba(254,205,211,0.9);
        border-radius: 24px;
        padding: 30px 34px;
        margin-bottom: 30px;
        box-shadow: 0 12px 35px rgba(15,23,42,0.07);
        backdrop-filter: blur(7px);
    }}

    .hero-row {{
        display: flex;
        align-items: center;
        gap: 20px;
    }}

    .hero-icon {{
        font-size: 48px;
        line-height: 1;
    }}

    .hero-title {{
        color: #991b1b;
        font-size: 36px;
        font-weight: 800;
        margin: 0;
        line-height: 1.2;
    }}

    .hero-subtitle {{
        color: #64748b;
        font-size: 15px;
        margin-top: 9px;
        line-height: 1.6;
    }}

       SECTION HEADERS

    .section-header {{
        display: flex;
        align-items: center;
        gap: 14px;
        margin-top: 12px;
        margin-bottom: 18px;
    }}

    .section-icon {{
        width: 46px;
        height: 46px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 14px;
        background: #f1eafe;
        font-size: 25px;
        flex-shrink: 0;
    }}

    .section-title {{
        color: #172033;
        font-size: 24px;
        font-weight: 800;
        margin: 0;
        line-height: 1.2;
    }}

    .section-subtitle {{
        color: #7b8494;
        font-size: 14px;
        margin: 5px 0 0;
        line-height: 1.5;
    }}

       CARDS

    .metric-card {{
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(226,232,240,0.95);
        border-radius: 18px;
        padding: 22px;
        min-height: 112px;
        box-shadow: 0 8px 26px rgba(15,23,42,0.055);
        backdrop-filter: blur(5px);
    }}

    .metric-label {{
        color: #64748b;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    .metric-value {{
        color: #172033;
        font-size: 21px;
        font-weight: 800;
        margin-top: 10px;
        line-height: 1.25;
    }}

    .info-card {{
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(226,232,240,0.95);
        border-radius: 20px;
        padding: 25px;
        min-height: 250px;
        box-shadow: 0 8px 26px rgba(15,23,42,0.055);
        backdrop-filter: blur(5px);
    }}

    .info-card-title {{
        color: #172033;
        font-size: 20px;
        font-weight: 800;
        margin-bottom: 13px;
    }}

    .info-card-text {{
        color: #536174;
        font-size: 14px;
        line-height: 1.8;
    }}

    .info-step {{
        display: flex;
        gap: 12px;
        margin: 12px 0;
        align-items: flex-start;
    }}

    .info-step-number {{
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: #fce7f3;
        color: #be185d;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 800;
        flex-shrink: 0;
    }}

    .privacy-card {{
        background: linear-gradient(
            135deg,
            rgba(239,246,255,0.92),
            rgba(248,250,252,0.92)
        );
        border: 1px solid #dbeafe;
        border-radius: 18px;
        padding: 22px;
        margin-top: 18px;
    }}

    .privacy-title {{
        color: #1e3a8a;
        font-size: 17px;
        font-weight: 800;
        margin-bottom: 7px;
    }}

    .privacy-text {{
        color: #475569;
        font-size: 13px;
        line-height: 1.7;
    }}

       RESULT CARDS

    .result-positive {{
        background: linear-gradient(
            135deg,
            rgba(255,241,242,0.97),
            rgba(255,247,248,0.94)
        );
        border: 1px solid #fecdd3;
        border-radius: 22px;
        padding: 30px;
        box-shadow: 0 10px 30px rgba(244,63,94,0.09);
    }}

    .result-negative {{
        background: linear-gradient(
            135deg,
            rgba(236,253,245,0.97),
            rgba(247,255,251,0.94)
        );
        border: 1px solid #a7f3d0;
        border-radius: 22px;
        padding: 30px;
        box-shadow: 0 10px 30px rgba(16,185,129,0.09);
    }}

    .result-title {{
        color: #172033;
        font-size: 27px;
        font-weight: 800;
        line-height: 1.3;
    }}

    .result-number {{
        color: #172033;
        font-size: 43px;
        font-weight: 800;
        margin-top: 9px;
    }}

    .result-description {{
        color: #64748b;
        font-size: 14px;
        margin-top: 6px;
        line-height: 1.6;
    }}

    .disclaimer-card {{
        background: rgba(255,251,235,0.95);
        border: 1px solid #fde68a;
        border-radius: 17px;
        padding: 19px 22px;
        margin-top: 22px;
    }}

    .disclaimer-title {{
        color: #92400e;
        font-size: 15px;
        font-weight: 800;
        margin-bottom: 5px;
    }}

    .disclaimer-text {{
        color: #78350f;
        font-size: 13px;
        line-height: 1.65;
    }}

       FORM ELEMENTS

    div[data-testid="stNumberInput"],
    div[data-testid="stSelectbox"] {{
        margin-bottom: 8px;
    }}

    label {{
        font-weight: 600 !important;
        color: #334155 !important;
    }}

    .field-help {{
        color: #7b8494;
        font-size: 11px;
        margin-top: -4px;
        margin-bottom: 10px;
    }}

    /* Primary button */
    .stButton > button {{
        border-radius: 14px;
        min-height: 52px;
        font-weight: 800;
        font-size: 15px;
        border: none;
        box-shadow: 0 7px 18px rgba(225,29,72,0.14);
    }}

    /* Download button */
    .stDownloadButton > button {{
        border-radius: 13px;
        min-height: 48px;
        font-weight: 700;
    }}

       FOOTER

    .footer {{
        text-align: center;
        color: #94a3b8;
        font-size: 11px;
        padding: 35px 0 10px;
        line-height: 1.7;
    }}

       RESPONSIVE

    @media (max-width: 900px) {{
        [data-testid="stAppViewContainer"] .main .block-container {{
            padding-top: 2.5rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }}

        .hero-title {{
            font-size: 28px;
        }}

        .hero {{
            padding: 24px;
        }}

        section[data-testid="stSidebar"] {{
            min-width: 280px;
            max-width: 280px;
        }}
    }}

    </style>
    """,
    unsafe_allow_html=True,
)

# SESSION STATE

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# SIDEBAR

with st.sidebar:

    render_markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-heart">❤️</div>
            <div class="brand-title">HealthAI</div>
            <div class="brand-subtitle">
                Heart Health Assessment
            </div>
        </div>

        <div class="sidebar-divider"></div>

        <div class="sidebar-section-title">
            Main Menu
        </div>
        """,
        unsafe_allow_html=True,
    )

    navigation_options = [
        "🏠 Dashboard",
        "❤️ Risk Assessment",
        "📋 My Results",
    ]

    if "page" not in st.session_state:
        st.session_state.page = "🏠 Dashboard"

    page = st.radio(
        "Navigation",
        navigation_options,
        index=navigation_options.index(st.session_state.page),
        label_visibility="collapsed",
    )

    st.session_state.page = page

    render_markdown(
        """
        <div class="sidebar-divider"></div>

        <div class="sidebar-bottom">
            <div class="sidebar-bottom-heart">💗</div>
            Your heart health companion
            <br>
            Simple. Private. Easy to understand.
        </div>
        """,
        unsafe_allow_html=True,
    )

# DASHBOARD

if page == "🏠 Dashboard":

    render_markdown(
        """
        <div class="hero">
            <div class="hero-row">
                <div class="hero-icon">❤️</div>
                <div>
                    <div class="hero-title">
                        Heart Health Assessment
                    </div>
                    <div class="hero-subtitle">
                        A simple way to understand your estimated heart
                        disease risk using the health information you provide.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_markdown(
        """
        <div class="section-header">
            <div class="section-icon">🏠</div>
            <div>
                <div class="section-title">Welcome</div>
                <div class="section-subtitle">
                    Get started with a quick heart health assessment.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        render_markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Step 01</div>
                <div class="metric-value">Your Information</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        render_markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Step 02</div>
                <div class="metric-value">Health Details</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        render_markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Step 03</div>
                <div class="metric-value">Your Assessment</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    left, right = st.columns([1.35, 1])

    with left:
        render_markdown(
            """
            <div class="info-card">
                <div class="info-card-title">
                    💗 How it works
                </div>

                <div class="info-card-text">
                    Complete a short assessment using information such as
                    your age, blood pressure, cholesterol and other health
                    measurements.

                    <div class="info-step">
                        <div class="info-step-number">1</div>
                        <div>
                            Enter your basic information.
                        </div>
                    </div>

                    <div class="info-step">
                        <div class="info-step-number">2</div>
                        <div>
                            Enter the health measurements available to you.
                        </div>
                    </div>

                    <div class="info-step">
                        <div class="info-step-number">3</div>
                        <div>
                            Submit the assessment.
                        </div>
                    </div>

                    <div class="info-step">
                        <div class="info-step-number">4</div>
                        <div>
                            Review the estimated result.
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        render_markdown(
            """
            <div class="info-card">
                <div class="info-card-title">
                    🩺 Before you begin
                </div>

                <div class="info-card-text">
                    Keep your recent health measurements available if
                    possible. If a measurement has been taken by a
                    healthcare professional, use the value provided to you.

                    <br><br>

                    If you are experiencing severe or urgent symptoms,
                    seek immediate medical attention rather than relying
                    on this assessment.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_markdown(
        """
        <div class="privacy-card">
            <div class="privacy-title">🔒 Your information</div>
            <div class="privacy-text">
                Please avoid entering names, phone numbers, addresses,
                medical record numbers or other personally identifiable
                information into this assessment.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    if st.button(
        "❤️ Start Heart Health Assessment",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.page = "❤️ Risk Assessment"
        st.rerun()

# RISK ASSESSMENT

elif page == "❤️ Risk Assessment":

    render_markdown(
        """
        <div class="hero">
            <div class="hero-row">
                <div class="hero-icon">❤️</div>
                <div>
                    <div class="hero-title">
                        Heart Health Assessment
                    </div>
                    <div class="hero-subtitle">
                        Enter your health information below to receive
                        an estimated assessment.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # PATIENT INFORMATION

    render_markdown(
        """
        <div class="section-header">
            <div class="section-icon">👤</div>
            <div>
                <div class="section-title">About You</div>
                <div class="section-subtitle">
                    Basic information and chest-pain details.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=50,
            step=1,
        )

    with c2:
        sex = st.selectbox(
            "Sex",
            ["M", "F"],
            format_func=lambda x: "Male" if x == "M" else "Female",
        )

    with c3:
        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ATA", "NAP", "ASY", "TA"],
            format_func=lambda x: {
                "ATA": "Atypical Angina",
                "NAP": "Non-Anginal Pain",
                "ASY": "Asymptomatic",
                "TA": "Typical Angina",
            }[x],
        )

    render_markdown(
        "<div class='field-help'>Choose the option that best matches the information provided by your healthcare professional.</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # HEALTH MEASUREMENTS

    render_markdown(
        """
        <div class="section-header">
            <div class="section-icon">🩺</div>
            <div>
                <div class="section-title">Heart Health Details</div>
                <div class="section-subtitle">
                    Enter your blood pressure, cholesterol and heart-rate information.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        resting_bp = st.number_input(
            "Resting Blood Pressure",
            min_value=0,
            max_value=250,
            value=120,
            step=1,
            help="Resting systolic blood pressure, usually measured in mmHg.",
        )

    with c2:
        cholesterol = st.number_input(
            "Cholesterol",
            min_value=0,
            max_value=700,
            value=200,
            step=1,
            help="Cholesterol measurement from your blood test.",
        )

    with c3:
        max_hr = st.number_input(
            "Maximum Heart Rate",
            min_value=50,
            max_value=250,
            value=150,
            step=1,
            help="Maximum heart rate recorded during activity or an exercise test.",
        )

    st.divider()

    # MEDICAL INFORMATION

    render_markdown(
        """
        <div class="section-header">
            <div class="section-icon">🧬</div>
            <div>
                <div class="section-title">Medical Information</div>
                <div class="section-subtitle">
                    Additional health measurements used by the assessment.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        fasting_bs = st.selectbox(
            "Fasting Blood Sugar",
            [0, 1],
            format_func=lambda x:
                "Normal" if x == 0 else "High",
            help="Select the category indicated by your health report.",
        )

    with c2:
        resting_ecg = st.selectbox(
            "Resting ECG",
            ["Normal", "ST", "LVH"],
            help="Select the ECG category from your report.",
        )

    with c3:
        exercise_angina = st.selectbox(
            "Exercise-related Chest Discomfort",
            ["N", "Y"],
            format_func=lambda x:
                "No" if x == "N" else "Yes",
        )

    c1, c2 = st.columns(2)

    with c1:
        oldpeak = st.number_input(
            "ST Depression (Oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Use the value from your ECG or exercise-test report.",
        )

    with c2:
        st_slope = st.selectbox(
            "ST Segment Slope",
            ["Up", "Flat", "Down"],
            help="Use the category from your cardiac test report.",
        )

    st.write("")

    render_markdown(
        """
        <div class="privacy-card">
            <div class="privacy-title">💡 Please use accurate information</div>
            <div class="privacy-text">
                For measurements such as blood pressure, cholesterol and ECG
                results, use values from a recent and reliable health report
                whenever available.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    predict = st.button(
        "🔍 Assess My Heart Health",
        type="primary",
        use_container_width=True,
    )

    if predict:

        input_data = pd.DataFrame({
            "Age": [age],
            "Sex": [sex],
            "ChestPainType": [chest_pain],
            "RestingBP": [resting_bp],
            "Cholesterol": [cholesterol],
            "FastingBS": [fasting_bs],
            "RestingECG": [resting_ecg],
            "MaxHR": [max_hr],
            "ExerciseAngina": [exercise_angina],
            "Oldpeak": [oldpeak],
            "ST_Slope": [st_slope],
        })

        try:
            prediction = int(model.predict(input_data)[0])

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]
                classes = list(model.classes_)

                if 1 in classes:
                    positive_index = classes.index(1)
                    risk_probability = float(
                        probabilities[positive_index]
                    )
                else:
                    risk_probability = float(probabilities[-1])
            else:
                risk_probability = float(prediction)

            risk_percent = risk_probability * 100

            # Save result for My Results page.
            st.session_state.last_result = {
                "prediction": prediction,
                "risk_percent": risk_percent,
                "input_data": input_data.copy(),
            }

            st.divider()

            render_markdown(
                """
                <div class="section-header">
                    <div class="section-icon">📊</div>
                    <div>
                        <div class="section-title">Your Assessment</div>
                        <div class="section-subtitle">
                            Here is the result generated from the information you entered.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if prediction == 1:
                result_title = "⚠️ Higher Estimated Risk Class"
                result_class = "result-positive"
                result_text = (
                    "The assessment places this input in the positive "
                    "heart-disease class. This result is an estimate and "
                    "does not confirm that you have heart disease."
                )
            else:
                result_title = "✅ Lower Estimated Risk Class"
                result_class = "result-negative"
                result_text = (
                    "The assessment places this input in the negative "
                    "heart-disease class. This result does not guarantee "
                    "that heart disease is absent."
                )

            render_markdown(
                f"""
                <div class="{result_class}">
                    <div class="result-title">
                        {result_title}
                    </div>

                    <div class="result-number">
                        {risk_percent:.1f}%
                    </div>

                    <div class="result-description">
                        Estimated probability of the positive class
                        generated from the information entered.
                        <br><br>
                        {result_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            
            # GAUGE
            
            render_markdown("### 📈 Estimated Probability")

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_percent,
                    number={
                        "suffix": "%",
                        "font": {"size": 38},
                    },
                    title={
                        "text": "Estimated positive-class probability",
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100],
                        },
                        "bar": {
                            "color": "#e11d48",
                        },
                        "steps": [
                            {
                                "range": [0, 30],
                                "color": "#dcfce7",
                            },
                            {
                                "range": [30, 70],
                                "color": "#fef3c7",
                            },
                            {
                                "range": [70, 100],
                                "color": "#fee2e2",
                            },
                        ],
                    },
                )
            )

            gauge.update_layout(
                height=330,
                margin=dict(l=30, r=30, t=60, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#172033"),
            )

            st.plotly_chart(
                gauge,
                use_container_width=True,
            )

            # INPUT SUMMARY
            
            render_markdown("### 👤 Your Information")

            summary = pd.DataFrame({
                "Information": [
                    "Age",
                    "Sex",
                    "Chest Pain Type",
                    "Resting Blood Pressure",
                    "Cholesterol",
                    "Fasting Blood Sugar",
                    "Resting ECG",
                    "Maximum Heart Rate",
                    "Exercise-related Chest Discomfort",
                    "ST Depression",
                    "ST Segment Slope",
                ],
                "Value": [
                    age,
                    "Male" if sex == "M" else "Female",
                    {
                        "ATA": "Atypical Angina",
                        "NAP": "Non-Anginal Pain",
                        "ASY": "Asymptomatic",
                        "TA": "Typical Angina",
                    }[chest_pain],
                    resting_bp,
                    cholesterol,
                    "Normal" if fasting_bs == 0 else "High",
                    resting_ecg,
                    max_hr,
                    "No" if exercise_angina == "N" else "Yes",
                    oldpeak,
                    st_slope,
                ],
            })

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True,
            )
            
            # DOWNLOAD PDF REPORT
            
            pdf_data = create_pdf_report(
                input_data,
                prediction,
                risk_percent,
            )

            st.download_button(
                "⬇️ Download Assessment Report",
                data=pdf_data,
                file_name="heart_health_assessment_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

            render_markdown(
                """
                <div class="disclaimer-card">
                    <div class="disclaimer-title">
                        ⚕️ Important information
                    </div>

                    <div class="disclaimer-text">
                        This assessment provides an estimated result based
                        on the information entered. It is not a medical
                        diagnosis and should not replace advice, examination
                        or treatment from a qualified healthcare professional.
                        If you have concerning or emergency symptoms, seek
                        appropriate medical care immediately.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error("The assessment could not be completed.")
            st.code(str(e))

            st.info(
                "The loaded model must expect these exact original columns: "
                "Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, "
                "RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope."
            )

# MY RESULTS

elif page == "📋 My Results":

    render_markdown(
        """
        <div class="hero">
            <div class="hero-row">
                <div class="hero-icon">📋</div>
                <div>
                    <div class="hero-title">
                        My Assessment
                    </div>
                    <div class="hero-subtitle">
                        Review the latest heart health assessment completed
                        in this session.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    result = st.session_state.last_result

    if result is None:

        render_markdown(
            """
            <div class="info-card">
                <div class="info-card-title">
                    📋 No assessment yet
                </div>

                <div class="info-card-text">
                    You have not completed a heart health assessment in
                    this session.

                    <br><br>

                    Go to <b>Risk Assessment</b> from the sidebar to enter
                    your information and generate an assessment.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        prediction = result["prediction"]
        risk_percent = result["risk_percent"]

        if prediction == 1:
            result_title = "⚠️ Higher Estimated Risk Class"
            result_class = "result-positive"
            result_text = (
                "The assessment placed the entered information in the "
                "positive heart-disease class."
            )
        else:
            result_title = "✅ Lower Estimated Risk Class"
            result_class = "result-negative"
            result_text = (
                "The assessment placed the entered information in the "
                "negative heart-disease class."
            )

        render_markdown(
            f"""
            <div class="{result_class}">
                <div class="result-title">
                    {result_title}
                </div>

                <div class="result-number">
                    {risk_percent:.1f}%
                </div>

                <div class="result-description">
                    Estimated probability of the positive class.
                    <br><br>
                    {result_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        render_markdown("### 👤 Assessment Details")

        st.dataframe(
            result["input_data"],
            use_container_width=True,
            hide_index=True,
        )

        render_markdown(
            """
            <div class="disclaimer-card">
                <div class="disclaimer-title">
                    ⚕️ Remember
                </div>

                <div class="disclaimer-text">
                    This result is an estimate generated from the information
                    provided. It is not a diagnosis. For medical concerns,
                    consult a qualified healthcare professional.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# FOOTER

render_markdown(
    """
    <div class="footer">
        HealthPredict-AI • Heart Health Assessment<br>
        For informational and educational use
    </div>
    """,
    unsafe_allow_html=True,
)