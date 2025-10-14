from app.app import app

if __name__ == '__main__':
    print("=" * 60)
    print("🎵 MUSIC GENRE CLUSTERING DASHBOARD")
    print("=" * 60)
    print("\nStarting Flask server...")
    print("Dashboard will be available at: http://localhost:5000")
    print("\nPress CTRL+C to stop the server")
    print("=" * 60)




    app.run(debug=True, host='0.0.0.0', port=5000)
