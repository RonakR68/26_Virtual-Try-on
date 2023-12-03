// camera.js
async function getCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        const videoElement = document.createElement("video");
        videoElement.srcObject = stream;
        videoElement.play();
        const cameraContainer = document.getElementById("cameraContainer");
        cameraContainer.innerHTML = '';
        cameraContainer.appendChild(videoElement);
    } catch (error) {
        console.error("Error accessing camera:", error);
    }
}

// Add an event listener for the watch category buttons
document.getElementById("luxuryCategoryButton").addEventListener("click", () => {
    getCamera();
});

document.getElementById("analogCategoryButton").addEventListener("click", () => {
    getCamera();
});

document.getElementById("smartCategoryButton").addEventListener("click", () => {
    getCamera();
});

// Inline CSS styles for camera container and watch category buttons
const styles = `
.camera-container {
    width: 100%;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    position: relative;
    
     background-color: #333; /* Change the background color to your desired premium color */
    border: 4px solid #ffcc00; /* Add a rectangular border with the premium color */
    border-radius: 10px; /* Add rounded corners */
     overflow: hidden; /* Ensure the video stays within the container */
}

.camera-container video {
    max-width: 100%;
    max-height: 100%;
     width: 100%;
    height: 50%;
    transform: scaleX(-1); /* Flip video horizontally if needed */
}

.about-col a {
    color: #333;
    font-weight: bold;
    text-decoration: none;
    transition: color 0.3s;
}

.about-col a:hover {
    color: #007BFF;
}
`;

// Create a <style> element to inject the styles
const styleElement = document.createElement("style");
styleElement.innerHTML = styles;
document.head.appendChild(styleElement);

