
import math

# CSS3 / SVG Color Palette (BGR format for OpenCV)
# Source: Standard Web Colors
WEB_COLORS = {
    # Reds
    "Maroon": (0, 0, 128),
    "DarkRed": (0, 0, 139),
    "Brown": (42, 42, 165),
    "Firebrick": (34, 34, 178),
    "Crimson": (60, 20, 220),
    "Red": (0, 0, 255),
    "Tomato": (71, 99, 255),
    "Coral": (80, 127, 255),
    "IndianRed": (92, 92, 205),
    "LightCoral": (128, 128, 240),
    "DarkSalmon": (122, 150, 233),
    "Salmon": (114, 128, 250),
    "LightSalmon": (122, 160, 255),
    "OrangeRed": (0, 69, 255),
    "DarkOrange": (0, 140, 255),
    "Orange": (0, 165, 255),
    "Gold": (0, 215, 255),
    "DarkGoldenRod": (11, 134, 184),
    "GoldenRod": (32, 165, 218),
    "PaleGoldenRod": (170, 232, 238),
    "DarkKhaki": (107, 183, 189),
    "Khaki": (140, 230, 240),
    "Olive": (0, 128, 128),
    "Yellow": (0, 255, 255),
    "YellowGreen": (50, 205, 154),
    "DarkOliveGreen": (47, 107, 85),
    "OliveDrab": (35, 142, 107),
    "LawnGreen": (0, 252, 124),
    "Chartreuse": (0, 255, 127),
    "GreenYellow": (47, 255, 173),
    "DarkGreen": (0, 100, 0),
    "Green": (0, 128, 0),
    "ForestGreen": (34, 139, 34),
    "Lime": (0, 255, 0),
    "LimeGreen": (50, 205, 50),
    "PaleGreen": (152, 251, 152),
    "LightGreen": (144, 238, 144),
    "MediumSpringGreen": (154, 250, 0), # BGR check? 0, 250, 154 in RGB. (154, 250, 0) in BGR
    "SpringGreen": (127, 255, 0),
    "MediumSeaGreen": (113, 179, 60),
    "SeaGreen": (87, 139, 46),
    "Teal": (128, 128, 0),
    "DarkCyan": (139, 139, 0),
    "LightSeaGreen": (170, 178, 32),
    "CadetBlue": (160, 158, 95),
    "DarkTurquoise": (209, 206, 0),
    "MediumTurquoise": (204, 209, 72),
    "Turquoise": (208, 224, 64),
    "Aqua": (255, 255, 0),
    "Cyan": (255, 255, 0),
    "Aquamarine": (212, 255, 127),
    "PaleTurquoise": (238, 238, 175),
    "LightCyan": (255, 255, 224),
    "Navy": (128, 0, 0),
    "DarkBlue": (139, 0, 0),
    "MediumBlue": (205, 0, 0),
    "Blue": (255, 0, 0),
    "MidnightBlue": (112, 25, 25),
    "RoyalBlue": (225, 105, 65),
    "SteelBlue": (180, 130, 70),
    "DodgerBlue": (255, 144, 30),
    "DeepSkyBlue": (255, 191, 0),
    "CornflowerBlue": (237, 149, 100),
    "SkyBlue": (235, 206, 135),
    "LightSkyBlue": (250, 206, 135),
    "LightSteelBlue": (222, 196, 176),
    "LightBlue": (230, 216, 173),
    "PowderBlue": (230, 224, 176),
    "Indigo": (130, 0, 75),
    "Purple": (128, 0, 128),
    "DarkMagenta": (139, 0, 139),
    "DarkViolet": (211, 0, 148),
    "DarkOrchid": (204, 50, 153),
    "MediumOrchid": (211, 85, 186),
    "Magenta": (255, 0, 255),
    "Fuchsia": (255, 0, 255),
    "Violet": (238, 130, 238),
    "Plum": (221, 160, 221),
    "Thistle": (216, 191, 216),
    "Lavender": (250, 230, 230),
    "MistyRose": (225, 228, 255),
    "AntiqueWhite": (215, 235, 250),
    "Linen": (230, 240, 250),
    "Beige": (220, 245, 245),
    "WhiteSmoke": (245, 245, 245),
    "LavenderBlush": (245, 240, 255),
    "OldLace": (230, 245, 253),
    "AliceBlue": (255, 248, 240),
    "Seashell": (238, 245, 255),
    "GhostWhite": (255, 248, 248),
    "Honeydew": (240, 255, 240),
    "FloralWhite": (240, 250, 255),
    "Azure": (255, 255, 240),
    "MintCream": (250, 255, 245),
    "Snow": (250, 250, 255),
    "Ivory": (240, 255, 255),
    "White": (255, 255, 255),
    "Black": (0, 0, 0),
    "DarkSlateGray": (79, 79, 47),
    "DimGray": (105, 105, 105),
    "SlateGray": (144, 128, 112),
    "LightSlateGray": (153, 136, 119),
    "Gray": (128, 128, 128),
    "LightGray": (211, 211, 211),
    "DarkGray": (169, 169, 169),
    "Silver": (192, 192, 192),
    "Gainsboro": (220, 220, 220),
}
# Added custom 'Mint' based on recent finding
WEB_COLORS["Mint"] = (170, 255, 170)

def get_closest_color_name(rgb):
    """
    Input: rgb (B, G, R) tuple/list
    Output: Closest Name (String)
    """
    b, g, r = rgb
    min_dist = float("inf")
    best_name = "Unknown"
    
    for name, color in WEB_COLORS.items():
        # Euclidean Dist in BGR Space
        dist = (r - color[2])**2 + (g - color[1])**2 + (b - color[0])**2
        if dist < min_dist:
            min_dist = dist
            best_name = name
            
    return best_name
