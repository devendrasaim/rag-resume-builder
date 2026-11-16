# Launch RAG Resume Builder on Windows: backend (:8710) + frontend (:5180), then open the browser.
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONIOENCODING = "utf-8"
Start-Process -WindowStyle Minimized -FilePath "python" -ArgumentList "server.py" -WorkingDirectory "$root\backend"
Start-Process -WindowStyle Minimized -FilePath "npm.cmd" -ArgumentList "run", "dev" -WorkingDirectory "$root\frontend"
Start-Sleep -Seconds 4
Start-Process "http://localhost:5180"
