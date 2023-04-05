console.log("{{NAME}}")
console.log("{{DATASET}}")
console.log("{{SESSION_ID}}")
console.log({{PLAYBACK_ENABLED}})
console.log({{SAVE_ENABLED}})
console.log("{{SERVER_URL}}")

function createAudioElement(blobUrl) {
    const downloadEl = document.createElement('a');
    downloadEl.style = 'display: block';
    downloadEl.innerHTML = 'download';
    downloadEl.download = 'audio.webm';
    downloadEl.href = blobUrl;
    const audioEl = document.createElement('audio');
    audioEl.controls = true;
    const sourceEl = document.createElement('source');
    sourceEl.src = blobUrl;
    sourceEl.type = 'audio/webm';
    audioEl.appendChild(sourceEl);
    document.body.appendChild(audioEl);
    document.body.appendChild(downloadEl);

    return blobUrl
}

function record(){
    console.log('Recording...');
    navigator.mediaDevices.getUserMedia({ audio: true, video: false })
    .then(stream => {
        // Listening...
        let chunks = [];
        const recorder = new MediaRecorder(stream);
        recorder.ondataavailable = e => {
            chunks.push(e.data);
            console.log('Collected a chunk. State: ' + recorder.state);
            if (recorder.state == 'inactive') {
                console.log('Sending...');
                // convert stream data chunks to a 'webm' audio format as a blob
                const blob = new Blob(chunks, { type: 'audio/webm' });

                // Playback
                if ({{PLAYBACK_ENABLED}}) {
                    const blobUrl = URL.createObjectURL(blob);
                    console.log(blobUrl);
                    const audio = new Audio(blobUrl);
                    audio.play();
                }

                // Send it to the API
                const fd = new FormData();
                fd.append('name', '{{NAME}}');
                fd.append('dataset', '{{DATASET}}');
                fd.append('session_id', '{{SESSION_ID}}');
                fd.append('audio', blob, '{{NAME}}.webm');
                if ({{SAVE_ENABLED}} == 1){
                    fetch("{{SERVER_URL}}/save", {
                        method:"POST",
                        body: fd,
                        mode:'no-cors'
                    }).then(response => {
                            console.log(response)
                            if (response.ok || response.type === 'opaque') return response
                            else throw Error(`Server returned ${response.status}: ${response.statusText}`)
                    }).then(response => console.log(response.text()))
                    .catch(err => {
                        alert(err)
                    });
                }
            }
        };
        recorder.start(0);
        setTimeout(() => {
            recorder.stop()
        }, {{MILLISECONDS}});
    // Listening error
    }).catch(console.error);
}

record();

