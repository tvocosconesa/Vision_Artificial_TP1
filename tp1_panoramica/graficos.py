
from colorsys import hsv_to_rgb

import numpy as np 
from matplotlib import pyplot as plt
import cv2


def plot_matches(img1, kp1, img2, kp2, matches, titulo=None, figsize=(20, 10)):
    """Grafica cada match con un color distinto y circulos en sus keypoints."""
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, [], None,
        singlePointColor=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    for i, match in enumerate(matches):
        # El paso aureo separa los tonos de conexiones consecutivas.
        color = hsv_to_rgb((i * 0.618033988749895) % 1, 0.85, 1)
        x1, y1 = kp1[match.queryIdx].pt
        x2, y2 = kp2[match.trainIdx].pt
        x2 += img1.shape[1]
        plt.plot([x1, x2], [y1, y2], color=color, linewidth=1.2)
        plt.scatter(
            [x1, x2], [y1, y2], s=80, facecolors='none',
            edgecolors=[color], linewidths=2, zorder=3
        )
    plt.axis('off')
    plt.title(titulo if titulo else f'{len(matches)} matches encontrados')
    plt.show()


def plot_point_correspondences(img1, pts1, img2, pts2, titulo=None, figsize=(20, 10)):
    """Grafica pares de puntos seleccionados manualmente entre dos imagenes,
    numerando cada par y coloreandolo igual en ambos lados para poder
    identificar visualmente la correspondencia."""
    h1, w1 = img1.shape[:2]
    canvas = np.concatenate([
        cv2.cvtColor(img1, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    ], axis=1)

    plt.figure(figsize=figsize)
    plt.imshow(canvas)
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(pts1, pts2)):
        color = hsv_to_rgb((i * 0.618033988749895) % 1, 0.85, 1)
        x2_shifted = x2 + w1
        plt.plot([x1, x2_shifted], [y1, y2], color=color, linewidth=1.5, linestyle='--')
        plt.scatter([x1, x2_shifted], [y1, y2], s=150, facecolors='none',
                    edgecolors=[color], linewidths=2.5, zorder=3)
        plt.text(x1, y1 - 15, str(i + 1), color=color, fontsize=14, fontweight='bold', ha='center')
        plt.text(x2_shifted, y2 - 15, str(i + 1), color=color, fontsize=14, fontweight='bold', ha='center')
    plt.axis('off')
    plt.title(titulo if titulo else f'{len(pts1)} correspondencias seleccionadas')
    plt.show()


def plot_warp_result(img_src, img_dst, H, titulo=None, figsize=(12, 8)):
    """Aplica una homografia H a img_src y la mezcla (50/50) con img_dst
    para verificar visualmente que tan bien queda alineada la transformacion."""
    h, w = img_dst.shape[:2]
    warped = cv2.warpPerspective(img_src, H, (w, h))
    blend = cv2.addWeighted(warped, 0.5, img_dst, 0.5, 0)

    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(titulo if titulo else 'Resultado del warping (mezcla 50/50 con el ancla)')
    plt.show()

def plot_point_correspondences(img1, pts1, img2, pts2, titulo=None, figsize=(20, 10)):
    """Grafica pares de puntos seleccionados manualmente entre dos imagenes,
    numerando cada par y coloreandolo igual en ambos lados para poder
    identificar visualmente la correspondencia."""
    h1, w1 = img1.shape[:2]
    canvas = np.concatenate([
        cv2.cvtColor(img1, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    ], axis=1)

    plt.figure(figsize=figsize)
    plt.imshow(canvas)
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(pts1, pts2)):
        color = hsv_to_rgb((i * 0.618033988749895) % 1, 0.85, 1)
        x2_shifted = x2 + w1
        plt.plot([x1, x2_shifted], [y1, y2], color=color, linewidth=1.5, linestyle='--')
        
        # Círculos 
        plt.scatter([x1, x2_shifted], [y1, y2], s=60, facecolors='none',
                    edgecolors=[color], linewidths=1.5, zorder=3)
        
        # Números 
        plt.text(x1, y1 - 70, str(i + 1), color=color, fontsize=9, fontweight='bold', ha='center')
        plt.text(x2_shifted, y2 - 70, str(i + 1), color=color, fontsize=9, fontweight='bold', ha='center')
        
    plt.axis('off')
    plt.title(titulo if titulo else f'{len(pts1)} correspondencias seleccionadas')
    plt.show()


def plot_warp_result(img_src, img_dst, H, titulo=None,figsize=(12,6)):
    """Aplica una homografia H a img_src y la mezcla (50/50) con img_dst
    para verificar visualmente que tan bien queda alineada la transformacion."""
    h, w = img_dst.shape[:2]
    warped = cv2.warpPerspective(img_src, H, (w, h))
    blend = cv2.addWeighted(warped, 0.5, img_dst, 0.5, 0)
    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(titulo if titulo else 'Resultado del warping (mezcla 50/50 con el ancla)')
    plt.show()

def plot_warp_result_3(img_izq, img_centro, img_der, H_izq, H_der, titulo=None):
    """
    Aplica las homografías a las imágenes laterales y las mezcla de forma rápida
    (aprox. 33% cada una) con la imagen central usando el tamaño fijo del ancla.
    """
    h, w = img_centro.shape[:2]
    
    # Proyectar laterales al tamaño fijo de la imagen central
    warped_izq = cv2.warpPerspective(img_izq, H_izq, (w, h))
    warped_der = cv2.warpPerspective(img_der, H_der, (w, h))

    # Mezclar primero la izquierda con el centro (50/50)
    blend_parcial = cv2.addWeighted(warped_izq, 0.5, img_centro, 0.5, 0)
    # Mezclar el resultado con la derecha (ajustando pesos para que quede 33/33/33)
    blend_final = cv2.addWeighted(blend_parcial, 0.666, warped_der, 0.334, 0)

    plt.imshow(cv2.cvtColor(blend_final, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(titulo if titulo else 'Resultado warping 3 imágenes (mezcla simple)')
    plt.show()