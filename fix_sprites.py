import re

with open("sprites/wizard/generate_wizard_sprites.py", "r") as f:
    code = f.read()

# We want to add a neutral start and end frame to animations.
# For dance:
dance_fix = """
    # dance-start: neutral
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0, offset_x=0)
    save(c, "wizard-dance-0.png")

    # dance-1: shift right, bob up, hat -> purple
"""
code = code.replace("    # dance-1: shift right, bob up, hat → purple", dance_fix)

dance_end = """
    # dance-end: back to neutral
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0, offset_x=0)
    save(c, "wizard-dance-9.png")

    print()
"""
code = code.replace("    print()\n\n    # --- BOW FRAMES", dance_end + "    # --- BOW FRAMES")

# Apply similar neutral frames for others if they don't have it
code = re.sub(r'# --- ([A-Z]+) FRAMES \((.*?)\) ---\n\s*print\("=== .*? ===\"\)\n',
    lambda m: f'# --- {m.group(1)} FRAMES ({m.group(2)}) ---\n    print("=== {m.group(1).title()} Frames ===")\n    c = create_canvas()\n    draw_base_wizard(c, offset_y=0, offset_x=0)\n    save(c, "wizard-{m.group(1).lower()}-00.png")\n',
    code)

# And add end frames before print()
# Wait, this is tricky. Let's just manually patch the ones that matter.
with open("sprites/wizard/generate_wizard_sprites.py", "w") as f:
    f.write(code)
