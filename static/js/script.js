document.getElementById('predict-form').addEventListener('submit', function (e) {
    e.preventDefault();  // Prevent the default form submission
    
    let inputDays = document.querySelector('input[name="days"]').value;

    // Check if the input is a valid number
    if (isNaN(inputDays) || inputDays <= 0) {
        alert("Please enter a valid number of days.");
        return;
    }

    // Create form data to send via POST
    let formData = new FormData();
    formData.append('days', inputDays);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(data => {
        // Insert the response into the page
        document.querySelector('.result-container').innerHTML = data;
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while making the prediction.');
    });
});
