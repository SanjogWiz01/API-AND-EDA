// Import the Express module
const express = require('express');
const app = express();
const port = 3000;

app.use(express.json());

// Define a route for GET requests
app.get('/users', (req, res) => {
    res.json({ message: 'Returning list of users' });
});

// Define a route for POST requests
app.post('/users', (req, res) => {
    const newUser = req.body;
    res.json({ message: 'User created', user: newUser });
});

// Define a route for PUT requests
app.put('/users/:id', (req, res) => {
    const userId = req.params.id;
    const updatedUser = req.body;
    res.json({ message: `User with ID ${userId} updated`, updatedUser });
});

// Define a route for DELETE requests
app.delete('/users/:id', (req, res) => {
    const userId = req.params.id;
    res.json({ message: `User with ID ${userId} deleted` });
});

// Start the server
app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});