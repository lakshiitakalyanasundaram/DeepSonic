document.addEventListener('DOMContentLoaded', function() {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-upload');
    const analyzeButton = document.getElementById('analyze-button');
    const fileDetails = document.getElementById('file-details');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    const fileDuration = document.getElementById('file-duration');
    const waveform = document.getElementById('waveform');
    const resultContainer = document.querySelector('.result-container');
    const sampleSelector = document.getElementById('sample-selector');

    let currentFile = null;
    let wavesurfer = null;
    let isPlaying = false;

    // Initialize WaveSurfer
    function initWaveSurfer() {
        if (wavesurfer) {
            wavesurfer.destroy();
        }
        wavesurfer = WaveSurfer.create({
            container: '#waveform',
            waveColor: '#9d50bb',
            progressColor: '#6e48aa',
            cursorColor: '#333',
            barWidth: 2,
            barGap: 1,
            responsive: true,
            height: 100,
            backend: 'WebAudio'
        });

        // Add play/pause button
        const playButton = document.createElement('button');
        playButton.className = 'button-primary';
        playButton.style.marginTop = '10px';
        playButton.innerHTML = '<svg class="button-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M8 5v14l11-7z" fill="currentColor"/></svg> Play';
        
        // Remove existing play button if any
        const existingPlayButton = waveform.parentElement.querySelector('.play-button');
        if (existingPlayButton) {
            existingPlayButton.remove();
        }
        
        playButton.classList.add('play-button');
        waveform.parentElement.appendChild(playButton);

        playButton.addEventListener('click', function() {
            if (!wavesurfer.isPlaying()) {
                wavesurfer.play();
                playButton.innerHTML = '<svg class="button-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M6 4h4v16H6zm8 0h4v16h-4z" fill="currentColor"/></svg> Pause';
            } else {
                wavesurfer.pause();
                playButton.innerHTML = '<svg class="button-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M8 5v14l11-7z" fill="currentColor"/></svg> Play';
            }
        });

        // Update button when playback ends
        wavesurfer.on('finish', function() {
            playButton.innerHTML = '<svg class="button-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M8 5v14l11-7z" fill="currentColor"/></svg> Play';
        });

        return wavesurfer;
    }

    // Load sample files from server
    async function loadSampleFiles() {
        try {
            console.log('Fetching samples from server at /samples...');
            const response = await fetch('/samples');
            console.log('Server response status:', response.status);
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('Server response not OK:', errorText);
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
            }
            
            const samples = await response.json();
            console.log('Received samples from server:', samples);
            
            // Clear existing options except the first one
            console.log('Clearing existing options...');
            while (sampleSelector.options.length > 1) {
                sampleSelector.remove(1);
            }
            
            if (!Array.isArray(samples) || samples.length === 0) {
                console.log('No samples found or invalid response');
                const option = document.createElement('option');
                option.value = '';
                option.textContent = 'No sample files available';
                option.disabled = true;
                sampleSelector.appendChild(option);
                return;
            }
            
            // Add sample files to dropdown
            console.log(`Adding ${samples.length} samples to dropdown...`);
            samples.forEach((sample, index) => {
                console.log(`Adding sample ${index + 1}:`, sample);
                const option = document.createElement('option');
                option.value = sample.path;
                const duration = sample.duration ? ` (${formatDuration(sample.duration)})` : '';
                const size = ` - ${formatFileSize(sample.size)}`;
                option.textContent = `${sample.name}${duration}${size}`;
                sampleSelector.appendChild(option);
            });
            
            console.log('Sample selector update complete');
        } catch (error) {
            console.error('Error in loadSampleFiles:', error);
            // Add error message to dropdown
            while (sampleSelector.options.length > 1) {
                sampleSelector.remove(1);
            }
            const option = document.createElement('option');
            option.value = '';
            option.textContent = `Error loading samples: ${error.message}`;
            option.disabled = true;
            sampleSelector.appendChild(option);
        }
    }

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    // Handle dropped files
    dropArea.addEventListener('drop', handleDrop, false);
    dropArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFiles);

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight(e) {
        dropArea.classList.add('highlight');
    }

    function unhighlight(e) {
        dropArea.classList.remove('highlight');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles({ target: { files: files } });
    }

    function handleFiles(e) {
        const files = e.target.files;
        if (files.length === 0) return;

        const file = files[0];
        if (!file.name.toLowerCase().endsWith('.wav')) {
            alert('Please upload a WAV file');
            return;
        }

        currentFile = file;
        updateFileDetails(file);
        analyzeButton.disabled = false;
        sampleSelector.value = ''; // Reset sample selector
    }

    function updateFileDetails(file) {
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        fileDetails.style.display = 'block';

        // Create an audio element to get duration
        const audio = new Audio();
        audio.preload = 'metadata';
        audio.onloadedmetadata = function() {
            fileDuration.textContent = formatDuration(audio.duration);
        };
        audio.src = URL.createObjectURL(file);

        // Load waveform
        if (wavesurfer) {
            wavesurfer.load(URL.createObjectURL(file));
        }
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function formatDuration(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    // Handle sample selection
    sampleSelector.addEventListener('change', async function() {
        const selectedPath = this.value;
        if (!selectedPath) {
            fileDetails.style.display = 'none';
            analyzeButton.disabled = true;
            return;
        }

        try {
            // Fetch sample file details
            const response = await fetch(selectedPath);
            const blob = await response.blob();
            const file = new File([blob], selectedPath.split('/').pop(), { type: 'audio/wav' });
            
            currentFile = file;
            updateFileDetails(file);
            analyzeButton.disabled = false;

            // Load audio into WaveSurfer
            if (wavesurfer) {
                wavesurfer.load(URL.createObjectURL(file));
            }
        } catch (error) {
            console.error('Error loading sample:', error);
            alert('Error loading sample file');
        }
    });

    analyzeButton.addEventListener('click', async function() {
        if (!currentFile) return;

        analyzeButton.disabled = true;
        analyzeButton.innerHTML = '<span class="button-icon">⌛</span> Analyzing...';

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                displayResult(result);
            } else {
                throw new Error(result.error || 'Analysis failed');
            }
        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            analyzeButton.disabled = false;
            analyzeButton.innerHTML = '<svg class="button-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M9.5 16.5L14.5 12L9.5 7.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>Analyze Audio';
        }
    });

    function displayResult(result) {
        resultContainer.style.display = 'block';
        resultContainer.className = 'result-container ' + (result.prediction === 'Real' ? 'result-real' : 'result-fake');
        
        const confidence = (result.confidence * 100).toFixed(2);
        
        resultContainer.innerHTML = `
            <h3>${result.prediction}</h3>
            <div class="confidence">Confidence: ${confidence}%</div>
            <div class="progress-bar">
                <div class="progress-value" style="width: ${confidence}%; background-color: ${result.prediction === 'Real' ? '#28a745' : '#dc3545'}"></div>
            </div>
            <div style="margin-top: 20px;">
                <div>Real Probability: ${(result.probabilities.real * 100).toFixed(2)}%</div>
                <div>Fake Probability: ${(result.probabilities.fake * 100).toFixed(2)}%</div>
            </div>
        `;
    }

    // Initialize the application
    initWaveSurfer();
    loadSampleFiles();
}); 