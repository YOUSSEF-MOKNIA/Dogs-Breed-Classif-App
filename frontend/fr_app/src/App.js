import React, { useState } from 'react';
import ImageUpload from './components/ImageUpload';
import BreedDisplay from './components/BreedDisplay';
import './App.css';

function App() {
    const [prediction, setPrediction] = useState('');

    const handlePrediction = (predictedClass) => {
        setPrediction(predictedClass);
    };

    return (
        <div className="App">
            <h1>Dog Breed Classifier</h1>
            <ImageUpload onPrediction={handlePrediction} />
            {prediction && <BreedDisplay breed={prediction} />}
        </div>
    );
}

export default App;
