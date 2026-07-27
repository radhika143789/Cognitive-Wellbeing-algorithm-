document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultContainer = document.getElementById('result-container');
    const btnText = document.querySelector('.btn-text');
    const btnLoader = document.getElementById('btn-loader');
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // UI Loading state
        btnText.classList.add('hidden');
        btnLoader.classList.remove('hidden');
        const submitBtn = form.querySelector('button[type="submit"]');
        submitBtn.disabled = true;
        
        // Hide previous result if visible
        resultContainer.classList.remove('visible');
        
        // Gather data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            
            const result = await response.json();
            
            if (response.ok && result.success) {
                showResult(result.mental_fitness);
            } else {
                alert('Error: ' + (result.error || 'Failed to analyze data'));
            }
        } catch (error) {
            console.error('Error:', error);
            alert('A network error occurred while communicating with the server.');
        } finally {
            // Restore button state
            btnText.classList.remove('hidden');
            btnLoader.classList.add('hidden');
            submitBtn.disabled = false;
        }
    });
    
    function showResult(score) {
        resultContainer.classList.remove('hidden');
        
        // Force reflow
        void resultContainer.offsetWidth;
        
        resultContainer.classList.add('visible');
        
        // Animate circular chart
        const circle = document.querySelector('.circle');
        const text = document.querySelector('.percentage');
        const msg = document.getElementById('result-message');
        
        // Determine color and message based on score
        let color = '#ef4444'; // Red for low
        let message = 'Your cognitive well-being requires attention. Consider seeking professional guidance.';
        
        if (score >= 80) {
            color = '#10b981'; // Green for high
            message = 'Excellent cognitive well-being! Keep up your healthy lifestyle.';
        } else if (score >= 50) {
            color = '#f59e0b'; // Yellow for medium
            message = 'Moderate cognitive well-being. Small positive habits could improve your score.';
        }
        
        circle.style.stroke = color;
        
        // Animate the stroke dash array and number
        setTimeout(() => {
            circle.style.strokeDasharray = `${score}, 100`;
            animateValue(text, 0, score, 1500);
        }, 300);
        
        msg.textContent = message;
    }
    
    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = Math.floor(progress * (end - start) + start);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }
});
