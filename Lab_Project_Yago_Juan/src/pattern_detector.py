import cv2
from EI import find_largest_quad, warp_quad, draw_text, PatternDetector, StableLabel, SequenceDecoder
import time

def pattern_detector():
    detector = PatternDetector(inner_margin_ratio=0.10, debug=False)
    stabilizer = StableLabel(window_size=12, min_consensus=8)

    expected = ["LINE_H", "LINE_V", "LINE_D1", "CIRCLE"]
    decoder = SequenceDecoder(expected_sequence=expected, max_len=4, reset_on_block=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    last_event_label = None
    message = "Muestra: LINE_H -> LINE_V -> LINE_D1 -> CIRCLE"
    message_timer = 0

    print("Controles:")
    print("  q -> salir")
    print("  r -> reset secuencia")
    print("Secuencia esperada:", expected, "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        quad = find_largest_quad(frame, min_area_ratio=0.06)

        if quad is not None:
            cv2.polylines(frame, [quad.astype(int)], True, (0, 255, 0), 2)
            warped = warp_quad(frame, quad, out_size=600)

            label, score, _dbg = detector.detect(warped)
            if score < 0.55:
                label = "UNKNOWN"

            stable_label, count, stable = stabilizer.update(label)

            draw_text(frame, f"Now: {label} ({score:.2f})", (20, 40), scale=1.0)
            draw_text(frame,
                      f"Stable: {stable_label} ({count}/{stabilizer.window_size})" if stable
                      else f"Stable: --- ({count}/{stabilizer.window_size})",
                      (20, 80), scale=1.0)

            if stable and stable_label not in ("UNKNOWN", "NO_TARGET"):
                if stable_label != last_event_label:
                    last_event_label = stable_label
                    print(f"[EVENTO] {stable_label}")

                    status, buf = decoder.update(stable_label)

                    if status == "IN_PROGRESS":
                        message = f"Secuencia: {buf}"
                        message_timer = 50

                    elif status == "BLOCK":
                        message = "BLOCK ❌ Orden incorrecto. Reset."
                        message_timer = 90
                        print("[DECODER] BLOCK")

                    elif status == "ALLOW":
                        message = "ALLOW ✅ Secuencia correcta. Paso permitido."
                        message_timer = 120
                        print("[DECODER] ALLOW")
                        time.sleep(0.5)
                        break

            cv2.imshow("Rectified", warped)

        else:
            stabilizer.update("UNKNOWN")
            draw_text(frame, "NO_TARGET (ensenia una hoja rectangular)", (20, 40), scale=1.0)

        draw_text(frame, f"Buffer: {decoder.state()}", (20, 120), scale=0.9)

        if message_timer > 0:
            draw_text(frame, message, (20, 160), scale=0.9)
            message_timer -= 1

        cv2.imshow("Webcam - Detector + Decoder", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            decoder.reset()
            stabilizer.reset()
            last_event_label = None
            message = "Reset manual ✅"
            message_timer = 60
            print("[INFO] Reset manual.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    pattern_detector()
