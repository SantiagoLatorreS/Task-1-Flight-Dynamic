import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyBboxPatch
from math import atan2, sqrt, degrees

def body_to_ned_dcm(phi, theta, psi):
    """Calcula la Matriz de Cosenos Directores (DCM) de Body a NED"""
    φ, θ, ψ = np.deg2rad([phi, theta, psi])
    cφ, sφ = np.cos(φ), np.sin(φ)
    cθ, sθ = np.cos(θ), np.sin(θ)
    cψ, sψ = np.cos(ψ), np.sin(ψ)

    Rz = np.array([[cψ, -sψ, 0], [sψ, cψ, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, cφ, -sφ], [0, sφ, cφ]])
    Ry = np.array([[cθ, 0, sθ], [0, 1, 0], [-sθ, 0, cθ]])

    return Rz @ Ry @ Rx  # Secuencia aeronáutica ZYX (3-2-1)

def compute_angles(u, v, w):
    """Calcula el ángulo de ataque (α) y el de deslizamiento (β)"""
    V = sqrt(u*u + v*v + w*w)
    if V < 1e-6:
        return 0.0, 0.0
    α = degrees(atan2(w, u))
    β = degrees(atan2(v, sqrt(u*u + w*w)))
    return α, β

def flight_path_angle(vel_ned):
    """Calcula el ángulo de trayectoria de vuelo (γ) desde el vector NED"""
    V_N, V_E, V_D = vel_ned
    V_horiz = sqrt(V_N**2 + V_E**2)
    return degrees(atan2(-V_D, V_horiz))  # Negativo porque D+ = abajo

def aircraft_state(alpha, beta, gamma, u, v, w, u_ned, v_ned, w_ned, p, q, r, phi, theta, psi):
    """Retorna los valores de estado de la aeronave en un diccionario"""
    return {
        "angles": {"alpha": alpha, "beta": beta, "gamma": gamma},
        "velocities_body": np.array([u, v, w]),
        "velocities_ned": np.array([u_ned, v_ned, w_ned]),
        "angular_rates": np.array([p, q, r]),
        "attitude": np.array([phi, theta, psi])
    }

def set_axes_equal(ax):
    """Ajusta la escala de los ejes 3D para que no se deforme el gráfico"""
    x0,x1 = ax.get_xlim3d()
    y0,y1 = ax.get_ylim3d()
    z0,z1 = ax.get_zlim3d()
    mx = max(x1-x0, y1-y0, z1-z0)/2
    cx,cy,cz = (x0+x1)/2, (y0+y1)/2, (z0+z1)/2
    ax.set_xlim3d(cx-mx, cx+mx)
    ax.set_ylim3d(cy-mx, cy+mx)
    ax.set_zlim3d(cz-mx, cz+mx)

def plot_enhanced(case_name, state, vel_rel_body, wind_ned, dcm_b2n):
    """Genera la interfaz gráfica (HUD, Aeronave detallada, Marcos NED/Body)"""
    # Crear un diseño tipo panel (Grid)
    fig = plt.figure(constrained_layout=True, figsize=(14, 7))
    fig.canvas.manager.set_window_title(f"Flight Dynamics - {case_name}")
    gs = fig.add_gridspec(2, 2, width_ratios=[2.5, 1], height_ratios=[1, 1.2])

    # ==========================================
    # 1. SUBPLOT 3D: AERONAVE Y VECTORES
    # ==========================================
    ax3 = fig.add_subplot(gs[:, 0], projection='3d')
    ax3.set_title(f"Visualización 3D: {case_name}", fontsize=14, fontweight='bold')

    # Invertir eje Z para respetar la convención NED (Z apunta hacia ABAJO)
    ax3.invert_zaxis()

    # ── Geometría realista de aeronave comercial ──
    # Dimensiones base
    L  = 2.4   # largo total del fuselaje
    R  = 0.18  # radio del fuselaje
    ns = 1.0   # longitud del cono de nariz
    ws = 1.5   # semi-envergadura del ala
    wc = 0.55  # cuerda del ala en la raíz
    wt = 0.20  # cuerda del ala en la punta
    sw = 0.35  # flecha del ala (sweep hacia atrás)
    hs = 0.55  # semi-envergadura del estabilizador horizontal
    hc = 0.30  # cuerda del estabilizador horizontal
    vh = 0.55  # altura de la aleta vertical
    vc = 0.45  # cuerda de la aleta vertical
    er = 0.08  # radio de la góndola del motor
    el = 0.35  # largo de la góndola del motor

    # Posiciones clave en X (body frame, +x = nariz)
    x_nose  =  L / 2
    x_tail  = -L / 2
    x_wing  =  0.05          # borde de ataque del ala (raíz)
    x_htail = x_tail + 0.15  # borde de ataque del estabilizador
    x_vtail = x_tail + 0.10

    polys = []  # lista de (vértices, color_cara, color_borde)

    # ── FUSELAJE (prisma octogonal) ──
    n_sec = 8
    angles_sec = np.linspace(0, 2 * np.pi, n_sec, endpoint=False)
    # Secciones transversales en varias posiciones X
    x_stations = [x_nose, x_nose - 0.25, 0.4, 0.0, -0.4, x_tail + 0.35, x_tail]
    r_stations = [0.0,    R * 0.6,        R,   R,   R,    R * 0.75,      R * 0.35]

    sections = []
    for xs, rs in zip(x_stations, r_stations):
        ring = np.array([[xs, rs * np.cos(a), rs * np.sin(a)] for a in angles_sec])
        sections.append(ring)

    # Paneles laterales entre secciones consecutivas
    for i in range(len(sections) - 1):
        s0, s1 = sections[i], sections[i + 1]
        for j in range(n_sec):
            j1 = (j + 1) % n_sec
            quad = np.array([s0[j], s0[j1], s1[j1], s1[j]])
            # Parte superior blanca, inferior gris claro
            fc = '#F4F6F7' if j < n_sec // 2 else '#D5D8DC'
            polys.append((quad, fc, '#95A5A6'))

    # Tapar la punta de la nariz con triángulos
    tip = np.array([x_nose, 0, 0])
    s_first = sections[1]
    for j in range(n_sec):
        j1 = (j + 1) % n_sec
        tri = np.array([tip, s_first[j], s_first[j1]])
        polys.append((tri, '#F4F6F7', '#95A5A6'))

    # ── ALAS PRINCIPALES (con sweep y taper) ──
    for sign in [1, -1]:  # ala derecha e izquierda
        root_le = np.array([x_wing, 0, 0])
        root_te = np.array([x_wing - wc, 0, 0])
        tip_le  = np.array([x_wing - sw, sign * ws, -sign * 0.04])  # leve diedro
        tip_te  = np.array([x_wing - sw - wt, sign * ws, -sign * 0.04])
        quad = np.array([root_le, tip_le, tip_te, root_te])
        polys.append((quad, '#BDC3C7', '#7F8C8D'))

    # ── GÓNDOLAS DE MOTOR (bajo las alas) ──
    for sign in [1, -1]:
        eng_x = x_wing - sw * 0.45
        eng_y = sign * ws * 0.40
        eng_z = 0.12  # debajo del ala (Z+ = abajo en body)
        n_eng = 6
        a_eng = np.linspace(0, 2 * np.pi, n_eng, endpoint=False)
        front = np.array([[eng_x + el / 2, eng_y + er * np.cos(a), eng_z + er * np.sin(a)] for a in a_eng])
        back  = np.array([[eng_x - el / 2, eng_y + er * np.cos(a), eng_z + er * np.sin(a)] for a in a_eng])
        for j in range(n_eng):
            j1 = (j + 1) % n_eng
            quad = np.array([front[j], front[j1], back[j1], back[j]])
            polys.append((quad, '#5D6D7E', '#2C3E50'))
        # Tapas frontal y trasera
        polys.append((front, '#34495E', '#2C3E50'))
        polys.append((back,  '#85929E', '#2C3E50'))

    # Pilones motor→ala
    for sign in [1, -1]:
        eng_x = x_wing - sw * 0.45
        eng_y = sign * ws * 0.40
        pylon = np.array([
            [eng_x + el * 0.3, eng_y, 0],
            [eng_x - el * 0.3, eng_y, 0],
            [eng_x - el * 0.3, eng_y, 0.12 - er],
            [eng_x + el * 0.3, eng_y, 0.12 - er],
        ])
        polys.append((pylon, '#ABB2B9', '#7F8C8D'))

    # ── ESTABILIZADOR HORIZONTAL ──
    for sign in [1, -1]:
        root_le = np.array([x_htail, 0, 0])
        root_te = np.array([x_htail - hc, 0, 0])
        tip_le  = np.array([x_htail - 0.12, sign * hs, 0])
        tip_te  = np.array([x_htail - hc + 0.05, sign * hs, 0])
        quad = np.array([root_le, tip_le, tip_te, root_te])
        polys.append((quad, '#BDC3C7', '#7F8C8D'))

    # ── ALETA VERTICAL (con dorsal fillet) ──
    vtail = np.array([
        [x_vtail,          0, 0],            # base adelante
        [x_vtail - 0.12,   0, -vh],          # punta superior adelante
        [x_vtail - vc,     0, -vh * 0.85],   # punta superior atrás
        [x_vtail - vc - 0.05, 0, 0],         # base atrás
    ])
    polys.append((vtail, '#D5D8DC', '#7F8C8D'))

    # Pequeño timón (acento de color)
    rudder = np.array([
        [x_vtail - vc + 0.05,  0, -vh * 0.85],
        [x_vtail - vc - 0.05,  0, -vh * 0.85],
        [x_vtail - vc - 0.08,  0, -vh * 0.15],
        [x_vtail - vc + 0.02,  0, -vh * 0.15],
    ])
    polys.append((rudder, '#2E86C1', '#1B4F72'))

    # Franja de color en el fuselaje (ventanas / línea decorativa)
    stripe_pts = np.array([
        [0.5,  R * 0.97, -R * 0.15],
        [-0.7, R * 0.97, -R * 0.15],
        [-0.7, R * 0.97,  R * 0.05],
        [0.5,  R * 0.97,  R * 0.05],
    ])
    polys.append((stripe_pts, '#2E86C1', '#2E86C1'))
    stripe_pts_l = stripe_pts.copy()
    stripe_pts_l[:, 1] *= -1
    polys.append((stripe_pts_l, '#2E86C1', '#2E86C1'))

    # ── Dibujar todos los polígonos rotados ──
    for verts, fc, ec in polys:
        pts = (dcm_b2n @ verts.T).T
        poly = Poly3DCollection([pts], facecolor=fc, edgecolor=ec, alpha=0.85, linewidths=0.5)
        ax3.add_collection3d(poly)

    # Dibujar marco fijo NED estático (referencia)
    ax3.quiver(0,0,0, 1.5, 0, 0, color='gray', linestyle=':', arrow_length_ratio=0.1)
    ax3.quiver(0,0,0, 0, 1.5, 0, color='gray', linestyle=':', arrow_length_ratio=0.1)
    ax3.quiver(0,0,0, 0, 0, 1.5, color='gray', linestyle=':', arrow_length_ratio=0.1)
    ax3.text(1.6, 0, 0, 'N (North)', color='gray', fontsize=8)
    ax3.text(0, 1.6, 0, 'E (East)', color='gray', fontsize=8)
    ax3.text(0, 0, 1.6, 'D (Down)', color='gray', fontsize=8)

    # Dibujar ejes BODY rotados (Xb, Yb, Zb)
    for vec, col, lab in zip(np.eye(3), ['#E74C3C', '#27AE60', '#2980B9'], ['Xb (Roll)', 'Yb (Pitch)', 'Zb (Yaw)']):
        v = dcm_b2n @ vec
        ax3.quiver(0,0,0, *v, length=1.2, color=col, linewidth=2.5, arrow_length_ratio=0.15)
        ax3.text(*(v * 1.3), lab, color=col, fontsize=10, fontweight='bold')

    # Vectores de velocidad (Ajustados al marco global para visualización)
    vel_ned = state['velocities_ned']
    if np.linalg.norm(vel_ned) > 1e-6:
        v_ned_norm = vel_ned / np.linalg.norm(vel_ned)
        ax3.quiver(0,0,0, *v_ned_norm, length=1.2, color='#F39C12', linewidth=2)
        ax3.text(*(v_ned_norm * 1.3), 'V_ned', color='#F39C12', fontsize=10, fontweight='bold')

    # Convertir vel relativa (body) a NED para graficarla correctamente en el entorno 3D
    vel_rel_ned = dcm_b2n @ vel_rel_body
    if np.linalg.norm(vel_rel_ned) > 1e-6:
        v_rel_norm = vel_rel_ned / np.linalg.norm(vel_rel_ned)
        ax3.quiver(0,0,0, *v_rel_norm, length=1.2, color='#8E44AD', linestyle='--', linewidth=2)
        ax3.text(*(v_rel_norm * 1.3), 'V_rel', color='#8E44AD', fontsize=10, fontweight='bold')

    # Ajustes visuales de la gráfica 3D
    ax3.set_xlabel('North [m]')
    ax3.set_ylabel('East [m]')
    ax3.set_zlabel('Down [m]')
    ax3.auto_scale_xyz([-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5])
    set_axes_equal(ax3)

    # ==========================================
    # 2. SUBPLOT: ÁNGULOS AERODINÁMICOS (tarjetas de color)
    # ==========================================
    ax_angles = fig.add_subplot(gs[0, 1])
    ax_angles.axis('off')
    ax_angles.set_xlim(0, 1)
    ax_angles.set_ylim(0, 1)
    ax_angles.set_title("Ángulos Aerodinámicos", fontweight='bold', fontsize=13)

    angles = state['angles']
    cards = [
        ('α  Angle of Attack (AOA)',   angles['alpha'],  '#E74C3C'),  # Rojo
        ('β  Sideslip',                angles['beta'],   '#3498DB'),  # Azul
        ('γ  Flight Path',             angles['gamma'],  '#27AE60'),  # Verde
    ]

    card_h = 0.28        # altura de cada tarjeta
    gap    = 0.04        # espacio entre tarjetas
    y_start = 0.92       # posición superior

    for i, (label, val, color) in enumerate(cards):
        y = y_start - i * (card_h + gap)
        # Fondo de la tarjeta (rectángulo redondeado)
        fancy = FancyBboxPatch((0.05, y - card_h), 0.9, card_h,
                               transform=ax_angles.transAxes, clip_on=False,
                               facecolor=color, edgecolor='white',
                               linewidth=2, alpha=0.92,
                               boxstyle='round,pad=0.02')
        ax_angles.add_patch(fancy)
        # Nombre del ángulo
        ax_angles.text(0.5, y - card_h * 0.30, label,
                       transform=ax_angles.transAxes, ha='center', va='center',
                       fontsize=10, color='white', fontweight='bold', family='sans-serif')
        # Valor grande
        ax_angles.text(0.5, y - card_h * 0.70, f"{val:.2f}°",
                       transform=ax_angles.transAxes, ha='center', va='center',
                       fontsize=18, color='white', fontweight='bold', family='monospace')

    # ==========================================
    # 3. SUBPLOT TEXTO: PANEL DE INSTRUMENTOS
    # ==========================================
    ax_text = fig.add_subplot(gs[1, 1])
    ax_text.axis('off') # Ocultar ejes para usarlo solo como pantalla de texto

    # Extraer variables para el panel
    u, v, w = np.round(state['velocities_body'], 2)
    un, vn, wn = np.round(state['velocities_ned'], 2)
    N_w, E_w, D_w = np.round(wind_ned, 2)
    phi, theta, psi = np.round(np.rad2deg(state['attitude']), 2)
    mag_ned = np.round(np.linalg.norm(state['velocities_ned']), 2)
    mag_rel = np.round(np.linalg.norm(vel_rel_body), 2)

    # Texto formateado tipo tablero digital
    info_text = (
        f"--- FLIGHT INSTRUMENTS ---\n\n"
        f"ATTITUDE (Euler Angles):\n"
        f" Roll (φ): {phi}°\n"
        f" Pitch (θ): {theta}°\n"
        f" Yaw (ψ): {psi}°\n\n"
        f"VELOCITY VECTORS [m/s]:\n"
        f" Body  [u, v, w]: [{u}, {v}, {w}]\n"
        f" NED   [N, E, D]: [{un}, {vn}, {wn}]\n"
        f" Wind  [N, E, D]: [{N_w}, {E_w}, {D_w}]\n\n"
        f"MAGNITUDES [m/s]:\n"
        f" Ground Speed: {mag_ned}\n"
        f" True Airspeed: {mag_rel}"
    )

    # Caja de texto estilizada
    props = dict(boxstyle='round', facecolor='#2C3E50', alpha=0.9, edgecolor='#F1C40F')
    ax_text.text(0.05, 0.95, info_text, transform=ax_text.transAxes, fontsize=10,
                 color='white', verticalalignment='top', bbox=props, family='monospace')

    plt.show()

def process_case(case_name, u_kmh, v_kmh, w_kmh, phi, theta, psi, wind_ned_kmh):
    """Procesa y calcula la cinemática del caso de vuelo"""
    print(f"\n" + "="*50)
    print(f"=== {case_name} ===")

    # Convertir km/h a m/s
    vel_body = np.array([u_kmh, v_kmh, w_kmh]) * (1000 / 3600)
    wind_ned = np.array(wind_ned_kmh) * (1000 / 3600)

    # Cálculos y matrices
    dcm_b2n = body_to_ned_dcm(phi, theta, psi)
    wind_body = dcm_b2n.T @ wind_ned
    vel_rel_body = vel_body + wind_body
    vel_ned = dcm_b2n @ vel_body

    # Ángulos
    α, β = compute_angles(*vel_rel_body)
    γ = flight_path_angle(vel_ned)

    # Guardar en diccionario estructurado (tasas p,q,r = 0 por defecto)
    state = aircraft_state(
        alpha=α, beta=β, gamma=γ,
        u=vel_body[0], v=vel_body[1], w=vel_body[2],
        u_ned=vel_ned[0], v_ned=vel_ned[1], w_ned=vel_ned[2],
        p=0.0, q=0.0, r=0.0,
        phi=np.deg2rad(phi), theta=np.deg2rad(theta), psi=np.deg2rad(psi)
    )

    # Salida por terminal (requisito 4a de la asignación)
    print(f"\n  DCM (Body → NED):")
    for row in dcm_b2n:
        print(f"    [{row[0]:8.4f}  {row[1]:8.4f}  {row[2]:8.4f}]")
    print(f"\n  Velocity Body  [u, v, w] = [{vel_body[0]:.2f}, {vel_body[1]:.2f}, {vel_body[2]:.2f}] m/s")
    print(f"  Velocity NED   [N, E, D] = [{vel_ned[0]:.2f}, {vel_ned[1]:.2f}, {vel_ned[2]:.2f}] m/s")
    print(f"  Wind NED       [N, E, D] = [{wind_ned[0]:.2f}, {wind_ned[1]:.2f}, {wind_ned[2]:.2f}] m/s")
    print(f"  Vel. relativa  [u, v, w] = [{vel_rel_body[0]:.2f}, {vel_rel_body[1]:.2f}, {vel_rel_body[2]:.2f}] m/s")
    print(f"\n  Ángulos aerodinámicos:")
    print(f"    α (Angle of Attack) = {α:.2f}°")
    print(f"    β (Sideslip)        = {β:.2f}°")
    print(f"    γ (Flight Path)     = {γ:.2f}°")
    print(f"\n  Generando panel de visualización...")

    # Visualizar
    plot_enhanced(case_name, state, vel_rel_body, wind_ned, dcm_b2n)

def user_input_case():
    """Maneja la entrada manual del usuario"""
    try:
        u_kmh = float(input("u (forward km/h): "))
        v_kmh = float(input("v (lateral km/h): "))
        w_kmh = float(input("w (vertical km/h): "))

        φ = float(input("φ (roll deg): "))
        θ = float(input("θ (pitch deg): "))
        ψ = float(input("ψ (yaw deg): "))

        resp = input("¿Viento cruzado? (si/no): ").strip().lower()
        if resp in ('si', 's', 'yes', 'y'):
            N_kmh = float(input("  Viento North (km/h): "))
            E_kmh = float(input("  Viento East  (km/h): "))
            D_kmh = float(input("  Viento Down  (km/h): "))
            wind_ned_kmh = [N_kmh, E_kmh, D_kmh]
        else:
            wind_ned_kmh = [0, 0, 0]

        process_case("Entrada Manual de Usuario", u_kmh, v_kmh, w_kmh, φ, θ, ψ, wind_ned_kmh)
    except ValueError:
        print("\n[ERROR] Entrada inválida. Por favor, introduzca únicamente números.")

def predefined_cases():
    """Ejecuta los tres casos solicitados en la asignación"""
    cases = [
        {
            "name": "Case A: Straight & Level (No Wind)",
            "u_kmh": 180.0, "v_kmh": 0.0, "w_kmh": 0.0,
            "phi": 0.0, "theta": 0.0, "psi": 0.0,
            "wind_ned_kmh": [0.0, 0.0, 0.0]
        },
        {
            "name": "Case B: Climb / Descent",
            "u_kmh": 180.0, "v_kmh": 0.0, "w_kmh": -20.0,
            "phi": 0.0, "theta": 8.0, "psi": 0.0,
            "wind_ned_kmh": [10.0, 0.0, 0.0]
        },
        {
            "name": "Case C: Coordinated Turn",
            "u_kmh": 170.0, "v_kmh": 10.0, "w_kmh": 0.0,
            "phi": 25.0, "theta": 2.0, "psi": 15.0,
            "wind_ned_kmh": [0.0, 20.0, 0.0]
        }
    ]

    for case in cases:
        process_case(
            case["name"],
            case["u_kmh"], case["v_kmh"], case["w_kmh"],
            case["phi"], case["theta"], case["psi"],
            case["wind_ned_kmh"]
        )

def main():
    print("\n" + "*"*40)
    print(" AIRCRAFT COORDINATE FRAMES TOOL ".center(40, '*'))
    print("*"*40)

    while True:
        print("\nSeleccione el modo de operación:")
        print(" [1] Ingresar datos manualmente")
        print(" [2] Ejecutar Casos Predefinidos (A, B, C)")
        print(" [3] Salir")

        choice = input("\nOpción (1/2/3): ").strip()

        if choice == "1":
            user_input_case()
        elif choice == "2":
            predefined_cases()
        elif choice == "3":
            print("Cerrando el simulador... ¡Hasta luego!")
            break
        else:
            print("[ATENCIÓN] Opción inválida. Intente de nuevo.")

if __name__ == "__main__":
    main()