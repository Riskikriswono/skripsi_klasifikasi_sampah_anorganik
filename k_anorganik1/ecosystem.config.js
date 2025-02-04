module.exports = {
  apps: [
    {
      name: "klasifikasi_anorganik",
      script: ".venv/bin/python",
      args: "app.py --port 4444",
      interpreter: "none",
      cwd: "/home/riski/klasifikasi_anorganik",
      env: {
        FLASK_ENV: "development",
      },
      env_production: {
        FLASK_ENV: "production",
      },
    },
  ],
};
