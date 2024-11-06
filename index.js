const express = require('express');
const multer = require('multer');
const fs = require('fs');
const fetch = (...args) => import('node-fetch').then(({ default: fetch }) => fetch(...args));
const { exec } = require('child_process'); // Para ejecutar scripts de Python
const path = require('path'); // Para manejar rutas de archivos
const app = express();
const port = 3000;

// Configurar carpeta estática para servir index.html
app.use(express.static(path.join(__dirname, 'public')));

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
const datasetPath = path.join(__dirname, 'dataset');

function readDatabase() {
    const data = fs.readFileSync(databasePath, 'utf8');
    return JSON.parse(data).dataset;
}

async function downloadAudioFiles() {
    const datasetPath = path.join(__dirname, 'dataset');
    const driveFiles = readDatabase(); // Lee el archivo JSON

    for (const [folderName, folderContent] of Object.entries(driveFiles)) {
        const orgPath = path.join(datasetPath, folderName, 'org');
        fs.mkdirSync(orgPath, { recursive: true });

        for (const [fileName, fileUrl] of Object.entries(folderContent.org)) {
            const localFilePath = path.join(orgPath, fileName);

            // Solo descargar si el archivo no existe
            if (!fs.existsSync(localFilePath)) {
                const response = await fetch(fileUrl);

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


function getAudioCounts() {
    const result = {};

    // Recorre cada carpeta dentro de 'dataset'
    fs.readdirSync(datasetPath).forEach((folder) => {
        const folderPath = path.join(datasetPath, folder);
        const orgPath = path.join(folderPath, 'org');
        const cutPath = path.join(folderPath, 'cut');

        // Inicializa el conteo de archivos
        let orgCount = 0;
        let cutCount = 0;

        // Cuenta los archivos en la carpeta 'org' si existe
        if (fs.existsSync(orgPath)) {
            orgCount = fs.readdirSync(orgPath).filter(file => file.endsWith('.wav')).length;
        }

        // Cuenta los archivos en la carpeta 'cut' si existe
        if (fs.existsSync(cutPath)) {
            cutCount = fs.readdirSync(cutPath).filter(file => file.endsWith('.wav')).length;
        }

        // Almacena los conteos en el objeto de resultados
        result[folder] = {
            org: orgCount,
            cut: cutCount
        };
    });

    return result;
}

(async () => {
    await downloadAudioFiles();

    await app.get('/api/audios', (req, res) => {
        const audioCounts = getAudioCounts();
        res.json(audioCounts);
    });

    await app.post('/api/python', upload.single('audio'), (req, res) => {
        console.log(req.body)

        if (!req.file) {
            return res.json({ message: "No se ha cargado ningún archivo." });
        }

        const { username, password, slang } = req.body;

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

            const audioFilePath = req.file.path; // Ruta del archivo de audio
            const pythonScriptPath = path.join(__dirname, 'index.py');
            const pythonCommand = process.platform === "win32" ? "python" : "python3";

            // Ejecutar el script de Python
            exec(`${pythonCommand} "${pythonScriptPath}" "${audioFilePath}"`, (error, stdout, stderr) => {
                if (error) {
                    console.error(`Error ejecutando el script: ${error.message}`);
                    return res.json({ message: 'Error ejecutando el script de Python.' });
                }
                if (stderr) {
                    console.error(`Error en el script: ${stderr}`);
                    return res.json({ message: 'Error en el script de Python.' });
                }

                // Procesar la salida del script Python
                const response = JSON.parse(stdout);
                if (response.message.toLowerCase() == slang.toLowerCase()) {
                    res.json(response)
                } else {
                    res.json({ message: 'La frase no coincide.' })
                }

            });
        });
    });

    // Iniciar el servidor
    await app.listen(port, () => {
        console.log(`Servidor escuchando en http://localhost:${port}`);
    });
})();

