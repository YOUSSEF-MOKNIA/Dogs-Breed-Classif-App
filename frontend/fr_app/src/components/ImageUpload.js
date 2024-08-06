import React, { useState } from 'react';
import axios from 'axios';

const ImageUpload = ({ onPrediction }) => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewImage, setPreviewImage] = useState(null);
    const [error, setError] = useState(null);

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        setSelectedFile(file);
        setPreviewImage(URL.createObjectURL(file));
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await axios.post('http://localhost:8000/predict/', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            const { predicted_class, breed_image_path } = response.data;
            onPrediction(predicted_class, breed_image_path);
            setError(null);
        } catch (error) {
            console.error('Error uploading the file', error);
            setError('Error uploading the file. Please try again.');
        }
    };

    return (
        <div>
            <form onSubmit={handleSubmit}>
                <input type="file" onChange={handleFileChange} />
                <button type="submit">Upload</button>
            </form>
            {previewImage && (
                <div>
                    <h2>Selected Image Preview:</h2>
                    <img src={previewImage} alt="Selected for upload" style={{ maxWidth: '240px' }} />
                </div>
            )}
            {error && <p>{error}</p>}
        </div>
    );
};

export default ImageUpload;
