document.getElementById('uploadForm').onsubmit = async function(e) {
    e.preventDefault();

    const formData = new FormData();
    const fileField = document.getElementById('audioFile');

    if (fileField.files.length === 0) {
        showError('Please select a file to upload');
        return;
    }

    formData.append('file', fileField.files[0]);

    try {
        const response  = await fetch ('/predict', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();

        if (result.error) {
            showError(result.error);
            return;
        }

        //Display results
        document.getElementById('predictedGenre').textContent = result.predicted_genre;
        document.getElementById('confidence').textContent = result.confidence.toFixed(2);

        //Display top 3 predictions
        const top3List = document.getElementById('top3List');
        top3List.innerHTML = '';
        result.top_3_predictions.forEach(pred => {
            const li = document.createElement('li');
            li.textContent = `${pred.genre}: ${pred.confidence.toFixed(2)}%`;
            top3List.appendChild(li);
        });

        document.getElementById('result').style.display = 'block';
        document.getElementById('error').style.display = 'none';
    } 
    catch (error) {
        showError("An error occurred while processing the file");
    }

    function showError(message) {
        const errorDiv = document.getElementById('error');
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
        document.getElementById('result').style.display = 'none';
    }
}