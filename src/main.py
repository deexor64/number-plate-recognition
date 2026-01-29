from detect import detect_plates, save_results, show_results

def main():
    # Run detection and preprocessing
    print("Log: Begin detection")

    results = detect_plates("images/3cars.jpg", show_preprocessing=False)

    if results is None:
        return

    # Display results summary
    print(f"\n📊 SUMMARY")
    print("-" * 20)
    print(f"Plates detected: {len(results['detections'])}")
    print(f"Plates processed: {len([p for p in results['processed_plates'] if p])}")

    for i, detection in enumerate(results["detections"]):
        x, y, w, h = detection["coords"]
        conf = detection["confidence"]
        print(f"  Plate {i + 1}: ({x}, {y}) {w}×{h} confidence={conf:.3f}")

    # Show visual results
    show_results(results)

    # Optional: save results
    save_choice = input("\nSave results? (y/n): ").lower().startswith("y")
    if save_choice:
        save_results(results)

    print("✅ Complete!")


if __name__ == "__main__":
    main()
