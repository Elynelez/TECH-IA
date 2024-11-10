const express = require('express');
const multer = require('multer');
const fs = require('fs');
const fetch = (...args) => import('node-fetch').then(({ default: fetch }) => fetch(...args));
const { exec } = require('child_process'); // Para ejecutar scripts de Python
const path = require('path'); // Para manejar rutas de archivos
const app = express();
const port = 3000;
const { createProxyMiddleware } = require('http-proxy-middleware');

// Configurar carpeta estática para servir index.html
app.use(express.static(path.join(__dirname, 'public')));

app.use((req, res, next) => {
    console.log(`Request URL: ${req.url}`);
    next();
});

async function checkPythonServerReady(url, maxAttempts = 10, interval = 1000) {
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            const response = await fetch(url);
            if (response.ok) {
                console.log('Servidor de Python está listo.');
                return true;
            }
        } catch (error) {
            console.log(`Intento ${attempt}: El servidor de Python aún no está listo. Reintentando en ${interval}ms...`);
            await new Promise(resolve => setTimeout(resolve, interval));
        }
    }
    console.error('El servidor de Python no respondió después de varios intentos.');
    return false;
}

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/'); // Ruta donde se guardarán los archivos subidos
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + path.extname(file.originalname)); // Nombre único para el archivo
    },
});

const upload = multer({ storage });
const databasePath = path.join(__dirname, 'public/database.json');

function readDatabase() {
    const data = fs.readFileSync(databasePath, 'utf8');
    return JSON.parse(data).dataset;
}

async function startPythonAndInitializeProxy() {
    exec('python index.py', (error, stdout, stderr) => {
        if (error) {
            console.error(`Error al ejecutar el script de Python: ${error.message}`);
            return;
        }
        if (stderr) {
            console.error(`Error en el script de Python: ${stderr}`);
            return;
        }
        console.log(`Salida del script de Python:\n${stdout}`);
    });

    const serverReady = await checkPythonServerReady('http://localhost:5000/popo');
    if (serverReady) {
        app.use(createProxyMiddleware(['/popo', '/predict'], { target: "http://localhost:5000", "secure": "false" }));
    }
}

async function downloadAudioFiles() {
    const datasetPath = path.join(__dirname, 'dataset');
    const driveFiles = readDatabase(); // Lee el archivo JSON

    for (const [folderName, folderContent] of Object.entries(driveFiles)) {
        const orgPath = path.join(datasetPath, folderName, 'org');
        fs.mkdirSync(orgPath, { recursive: true });

        for (const [fileName, fileId] of Object.entries(folderContent.org)) {
            const localFilePath = path.join(orgPath, fileName);

            // Solo descargar si el archivo no existe
            if (!fs.existsSync(localFilePath)) {
                const url = `https://drive.google.com/uc?export=download&id=${fileId}`;
                const response = await fetch(url);

                if (response.ok) {
                    const fileStream = fs.createWriteStream(localFilePath);
                    response.body.pipe(fileStream);
                    console.log(`Descargado: ${fileName} en ${orgPath}`);
                } else {
                    console.error(`Error al descargar ${fileName}: ${response.statusText}`);
                }
            }
        }
    }
}

(async () => {
    await downloadAudioFiles();
    await startPythonAndInitializeProxy();

    app.post('/filter', upload.single('audio'), (req, res) => {
        console.log(req.body)

        if (!req.file) {
            return res.json({ message: "No se ha cargado ningún archivo." });
        }

        const { username, password } = req.body;

        // Leer el archivo database.json
        fs.readFile(databasePath, 'utf8', (err, data) => {
            if (err) {
                console.error('Error al leer el archivo de base de datos:', err);
                return res.json({ message: "error al leer la base de datos" })
            }

            // Parsear el contenido JSON
            const usersData = JSON.parse(data);
            const user = usersData.users.find(user => user.user === username && user.password === password);

            // Comprobar si el usuario existe y la contraseña es correcta
            if (!user) {
                return res.json({ message: 'Usuario o contraseña incorrectos.' });
            }
        });

        const audioFilePath = req.file.path

        return res.json({message: audioFilePath})
    });

    // Iniciar el servidor
    await app.listen(port, () => {
        console.log(`Servidor escuchando en http://localhost:${port}`);
    });
})();

