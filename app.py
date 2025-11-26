
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

# Configuración de la figura
fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=100)
plt.subplots_adjust(wspace=0.15, hspace=0.3)

def draw_mockup(ax, style_name, colors, font_color):
    # Configuración base del plot
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis('off')
    
    # 1. Fondo Principal
    ax.add_patch(patches.Rectangle((0, 0), 100, 60, color=colors['bg'], zorder=0))
    
    # 2. Barra Lateral (Sidebar)
    sidebar_width = 25
    ax.add_patch(patches.Rectangle((0, 0), sidebar_width, 60, color=colors['sidebar_bg'], zorder=1))
    
    # Elementos del Sidebar (Logo simulado)
    ax.add_patch(patches.Circle((sidebar_width/2, 52), 4, color=colors['accent'], alpha=0.8, zorder=2))
    ax.text(sidebar_width/2, 52, "$", color=colors.get('sidebar_text_contrast', 'white'), 
            ha='center', va='center', fontsize=14, fontweight='bold', zorder=3)
    
    # Menu Items simulados
    menu_y = 40
    for item in ["Inicio", "Análisis", "Predicción"]:
        # Icono
        ax.add_patch(patches.Circle((4, menu_y), 1.5, color=colors['accent'], alpha=0.5, zorder=2))
        # Texto
        ax.text(8, menu_y, item, color=colors['sidebar_text'], fontsize=9, va='center', zorder=2)
        menu_y -= 6

    # 3. Contenido Principal
    main_x = sidebar_width + 5
    
    # Títulos
    ax.text(main_x, 54, "Sistema Cash4Life", color=font_color, fontsize=12, fontweight='bold', zorder=2)
    ax.text(main_x, 50, "Investigación UPAO", color=colors['accent'], fontsize=8, zorder=2)
    
    # Tarjeta 1 (Info)
    card1_y = 30
    box1 = FancyBboxPatch((main_x, card1_y), 65, 12, boxstyle="round,pad=1", 
                          fc=colors['card1_bg'], ec=colors['card1_border'], lw=2, zorder=1)
    ax.add_patch(box1)
    ax.text(main_x + 2, card1_y + 8, "Planteamiento", color=colors['card1_text'], fontsize=9, fontweight='bold', zorder=2)
    ax.text(main_x + 2, card1_y + 4, "¿Existe sesgo en los\nnúmeros ganadores?", 
            color=colors['card1_text'], fontsize=8, zorder=2)

    # Tarjeta 2 (Success)
    card2_y = 10
    box2 = FancyBboxPatch((main_x, card2_y), 65, 12, boxstyle="round,pad=1", 
                          fc=colors['card2_bg'], ec=colors['card2_border'], lw=2, zorder=1)
    ax.add_patch(box2)
    ax.text(main_x + 2, card2_y + 8, "Metodología", color=colors['card2_text'], fontsize=9, fontweight='bold', zorder=2)
    ax.text(main_x + 2, card2_y + 4, "Uso de Regresión Lineal\ny Árboles de Decisión.", 
            color=colors['card2_text'], fontsize=8, zorder=2)

    # Etiqueta del estilo
    ax.text(50, -5, style_name, ha='center', fontsize=11, fontweight='bold', color='#333333')

# --- DEFINICIÓN DE COLORES ---

# 1. Académico Moderno
colors_academic = {
    'bg': '#F8F9FA', 'sidebar_bg': '#FFFFFF', 'sidebar_text': '#31333F',
    'accent': '#007BFF', 'card1_bg': '#E8F4F8', 'card1_border': '#007BFF',
    'card1_text': '#004085', 'card2_bg': '#D4EDDA', 'card2_border': '#28A745', 'card2_text': '#155724'
}
draw_mockup(axs[0, 0], "1. Académico Moderno", colors_academic, '#2C3E50')

# 2. Dark Mode Financiero
colors_dark = {
    'bg': '#0E1117', 'sidebar_bg': '#262730', 'sidebar_text': '#FAFAFA', 'sidebar_text_contrast': 'black',
    'accent': '#00C896', 'card1_bg': '#1E1E1E', 'card1_border': '#FF4B4B',
    'card1_text': '#FAFAFA', 'card2_bg': '#1E2A25', 'card2_border': '#00C896', 'card2_text': '#FAFAFA'
}
draw_mockup(axs[0, 1], "2. Dark Mode Tech", colors_dark, '#FFFFFF')

# 3. Estilo Editorial
colors_editorial = {
    'bg': '#FFFBF0', 'sidebar_bg': '#3E2723', 'sidebar_text': '#FFFBF0', 'sidebar_text_contrast': '#3E2723',
    'accent': '#D4AF37', 'card1_bg': '#F5F5F5', 'card1_border': '#3E2723',
    'card1_text': '#3E2723', 'card2_bg': '#E8F5E9', 'card2_border': '#2E7D32', 'card2_text': '#1B5E20'
}
draw_mockup(axs[1, 0], "3. Estilo Editorial", colors_editorial, '#3E2723')

# 4. Glassmorphism
colors_glass = {
    'bg': '#E0F7FA', 'sidebar_bg': '#FFFFFF', 'sidebar_text': '#455A64',
    'accent': '#00BCD4', 'card1_bg': '#FFFFFF', 'card1_border': '#B2EBF2',
    'card1_text': '#006064', 'card2_bg': '#FFFFFF', 'card2_border': '#B9F6CA', 'card2_text': '#1B5E20'
}
draw_mockup(axs[1, 1], "4. Glassmorphism", colors_glass, '#37474F')

plt.savefig('estilos_web_mockup.png', bbox_inches='tight')
