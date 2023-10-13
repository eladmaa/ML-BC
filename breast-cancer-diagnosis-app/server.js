const express = require('express');
const bodyParser = require('body-parser');
const session = require('express-session');
const fileUpload = require('express-fileupload');
const { spawn } = require('child_process');
const fs = require('fs');
const app = express();
app.use(express.static('public'));
const port = process.env.PORT || 3029;
const path_to_generated_model = 'saved_model.h5'
const path = require('path');
const { exec } = require('child_process');



// Configure session middleware
app.use(session({
    secret: 'your-secret-key', // Change this to a secure secret key
    resave: false,
    saveUninitialized: true,
}));

app.use(bodyParser.urlencoded({ extended: true }));
app.use(fileUpload());
app.use(express.static('public'));

// Middleware to check if user is authenticated
const requireAuth = (req, res, next) => {
    if (!req.session.isAuthenticated) {
        return res.redirect('/signin?message=Please sign in to access the app.');
    }
    next();
};



// Middleware for checking superuser authentication
const requireSuperuser = (req, res, next) => {
    if (req.session.isSuperuser) {
        next(); // User is a superuser, proceed to the next middleware
    } else {
        res.status(403).send('Access denied. Only superusers are allowed to train the model.');
    }
};

// Welcome page
app.get('/', (req, res) => {
    res.redirect('/welcome');
});

// Welcome page
app.get('/welcome', (req, res) => {
    res.sendFile(__dirname + '/public/index.html');
});

// Sign-in page
app.get('/signin', (req, res) => {
    const message = req.query.message || '';
    res.sendFile(__dirname + '/public/signin.html');
});

// Sign-in route
app.post('/signin', (req, res) => { 
    const { username, password } = req.body;

   
    if ((username === 'user' || username ==='superuser') && password === 'password') {
        req.session.isAuthenticated = true;

        // Check if the user is a superuser
        req.session.isSuperuser = (username === 'superuser');

        // Redirect superusers to the training page and regular users to the main app page
        if (req.session.isSuperuser) {
            res.sendFile(__dirname + '/public/train.html');
        } else {
            res.sendFile(__dirname + '/public/userAPP.html');
        }
    } else {
        // Redirect back to the sign-in page with an error message
        res.redirect('/signin?message=Incorrect username or password. Please try again.');
    }
});

// Main app page for regular users
app.get('/index', requireAuth, (req, res) => {
    const message = req.query.message || '';
    res.sendFile(__dirname + '/public/index.html');
});

// Training page for superusers
app.get('/train', requireAuth, requireSuperuser, (req, res) => {
    const message = req.query.message || '';
    res.sendFile(__dirname + '/public/train.html');
});

app.get('/performance', requireSuperuser, async (req, res) => {
    const performanceFilePath = path.join(__dirname, 'model', 'model_perform.json');
    try {
        const performData = await fs.promises.readFile(performanceFilePath, 'utf8');
        const performObj = JSON.parse(performData);

        // Extract the performance parameters from the JSON data
        res.json(performObj);
    } catch (err) {
        console.error(`Error reading model performance file: ${err}`);
        res.status(500).send(err.message);
    }
});

app.post('/trainModel', requireSuperuser, async (req, res) => {
    console.log('Received a request to /trainModel');

    // You can add a parameter 'modelFile' in the request body or query string to specify the model file
    const modelFile = req.body.modelFile || 'saved_model.h5';
    const epochs = req.body.epochs || 10;
    const batchSize = req.body.batchSize || 32;

    const pathMain = __dirname + '\\model\\main.py';
    const pathDataset = __dirname + '\\model\\METABRIC_RNA_Mutation.csv';
    const pathModel = __dirname + '\\model\\' + modelFile;

    // Execute the Python script with the specified model file, epochs, and batch size
    const util = require('util');
    const exec = util.promisify(require('child_process').exec);
    const command = `python ${pathMain} 2 ${pathDataset} ${pathModel} ${epochs} ${batchSize}`;

    try {
        const { stdout } = await exec(command);
        console.log('Python training script executed successfully');
        // console.log(stdout);

        const performFilePath = path.join(__dirname, 'model', 'model_perform.json');

        try {
            console.log(performFilePath);
            const performData = await fs.promises.readFile(performFilePath, 'utf8');
            const performObj = JSON.parse(performData);
        
            // Extract the accuracy or other performance parameters from the JSON data
            const accuracy = performObj.trainingAccuracy;
            console.log('Training Accuracy:', accuracy);
            // if ('showSaveFilePicker' in window) {
            //     // Request file system access
            //     const fileHandle = await showSaveFilePicker();
            
            //     // Write model data to the file
            //     try {
            //         const writable = await fileHandle.createWritable();
            //         await writable.write(modelData); // Replace with your model data
            //         await writable.close();
            //         console.log('Model saved successfully.');
            //     } catch (error) {
            //         console.error('Error writing to the file:', error);
            //     }
            // } else {
            //     // Handle unsupported browsers
            //     alert('File System Access API is not supported in this browser.');
            // }
            res.send(accuracy);
        } catch (err) {
            console.error(`Error reading performance parameters file: ${err.message}`);
            res.status(500).send(err.message);
        }
    } catch (error) {
        console.error('Error executing Python process:', error);
        res.status(500).send(`Error executing Python process: ${error.message}`);
    }
});

app.post('/executeScript', requireAuth, async (req, res) => {
    console.log('Received a request to /executeScript');
    if (!req.files || !req.files.geneVectorFile) {
        return res.status(400).send('No files were uploaded.');
    }

    const geneVectorFile = req.files.geneVectorFile;

    // Save the uploaded file temporarily (you can customize the path)
    const uploadPath = path.join(__dirname, 'uploads', geneVectorFile.name);

    geneVectorFile.mv(uploadPath, async (err) => {
        if (err) {
            return res.status(500).send(err);
        }

        const pathMain = __dirname + '\\model\\main.py';
        const pathUpload = __dirname + '\\uploads\\gene_vector.csv';

        // You can add a parameter 'modelFile' in the request body or query string to specify the model file
        const modelFile = req.body.modelFile || 'saved_model.h5';
        const pathModel = __dirname + '\\model\\' + modelFile;

        
        // Execute the Python script with the uploaded file, model file, and other parameters
        const util = require('util');
        const exec = util.promisify(require('child_process').exec);
        const command = `python ${pathMain} 1 ${pathUpload} ${pathModel}`;

        try {
            const { stdout } = await exec(command);
            console.log('Python script executed successfully');
            console.log(stdout);

            // Now you can read the predictions.json file and send it as a response
            // Assuming the predictions.json file is in the same directory as your Python script
            const predictionsFilePath = path.join(__dirname, 'model', 'predictions.json');

            try {
                const predictionsFilePath = path.join(__dirname, 'model', 'predictions.json');
                const modelPerformFilePath = path.join(__dirname, 'model', 'model_perform.json');

                
                // const predictionsData = await fs.promises.readFile(predictionsFilePath, 'utf8');
                console.log('Sending predictions as a response');

                const [predictionsData, modelPerformData] = await Promise.all([
                    fs.promises.readFile(predictionsFilePath, 'utf8'),
                    fs.promises.readFile(modelPerformFilePath, 'utf8'),
                ]);
                const predictions = JSON.parse(predictionsData);
                const modelPerformance = JSON.parse(modelPerformData);

                // Create an object to send both predictions and model performance
                const results = {
                    predictions,
                    modelPerformance,
                };
                res.send(results);
            } catch (err) {
                console.error(`Error reading predictions file: ${err}`);
                res.status(500).send(err);
            }
            
        } catch (error) {
            console.error('Error executing Python process:', error);
            res.status(500).send(`Error executing Python process: ${error.message}`);
        }
    });
});


app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
