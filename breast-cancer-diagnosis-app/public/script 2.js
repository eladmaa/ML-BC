// public/script.js
document.addEventListener('DOMContentLoaded', () => {
    const geneVectorFileInput = document.getElementById('gene-vector-file');
    const diagnoseButton = document.getElementById('diagnose-btn');
    const npiResultContainer = document.getElementById('result-container');
    const npiResultParagraph = document.getElementById('npi-result');

    geneVectorFileInput.addEventListener('change', () => {
        npiResultContainer.style.display = 'none';
    });

    diagnoseButton.addEventListener('click', () => {
        const file = geneVectorFileInput.files[0];
        if (!file) {
            alert('Please select a CSV file.');
            return;
        }

        const formData = new FormData();
        formData.append('geneVectorFile', file);

        fetch('/api/diagnose', {
            method: 'POST',
            body: formData,
        })
        .then((response) => response.json())
        .then((data) => {
            npiResultParagraph.textContent = `NPI: ${data.npi}`;
            npiResultContainer.style.display = 'block';
        })
        .catch((error) => console.error(error));
    });
});
