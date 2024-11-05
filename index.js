const express = require('express');
const multer = require('multer');
const fs = require('fs');
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

// Ruta simple que responde con un objeto JSON
app.get('/api/saludo', (req, res) => {
  res.json({
    mensaje: '¡Hola, este es un endpoint de prueba!',
    exito: true
  });
});

app.post('/api/python', upload.single('audio'), (req, res) => {
    console.log(req.body)

    if (!req.file) {
        return res.json({message: "No se ha cargado ningún archivo."});
    }

    const { username, password, slang } = req.body;

    // Leer el archivo database.json
    fs.readFile(databasePath, 'utf8', (err, data) => {
        if (err) {
            console.error('Error al leer el archivo de base de datos:', err);
            return res.json({message: "error al leer la base de datos"})
        }

        // Parsear el contenido JSON
        const usersData = JSON.parse(data);
        const user = usersData.users.find(user => user.user === username && user.password === password);

        // Comprobar si el usuario existe y la contraseña es correcta
        if (!user) {
            return res.json({message: 'Usuario o contraseña incorrectos.'});
        }

        const audioFilePath = req.file.path; // Ruta del archivo de audio
        const pythonScriptPath = path.join(__dirname, 'index.py');
    
        // Ejecutar el script de Python
        exec(`python "${pythonScriptPath}" "${audioFilePath}"`, (error, stdout, stderr) => {
            if (error) {
                console.error(`Error ejecutando el script: ${error.message}`);
                return res.status(500).send('Error ejecutando el script de Python.');
            }
            if (stderr) {
                console.error(`Error en el script: ${stderr}`);
                return res.status(500).send('Error en el script de Python.');
            }
    
            // Procesar la salida del script Python
            const response = JSON.parse(stdout);
            if(response.message.toLowerCase() == slang.toLowerCase()){
                res.json(response)
            } else {
                res.json({message: 'La frase no coincide.'})
            }
            
        });
    });
});

// Iniciar el servidor
app.listen(port, () => {
  console.log(`Servidor escuchando en http://localhost:${port}`);
});