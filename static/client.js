var pis = [
    {
        id: "rpi1",
        hostname: "192.168.50.128",
        port: 8888
    },
    {
        id: "rpi2",
        hostname: "192.168.50.128",
        port: 8888
    },
    {
        id: "rpi3",
        hostname: "192.168.50.128",
        port: 8888
    }
];

var pcs = {}

function negotiate(pi) {
    pcs[pi.id].addTransceiver('video', { direction: 'recvonly' });
    pcs[pi.id].addTransceiver('audio', { direction: 'recvonly' });
    return pcs[pi.id].createOffer().then(function (offer) {
        return pcs[pi.id].setLocalDescription(offer);
    }).then(function () {
        // wait for ICE gathering to complete
        return new Promise(function (resolve) {
            if (pcs[pi.id].iceGatheringState === 'complete') {
                resolve();
            } else {
                function checkState() {
                    if (pcs[pi.id].iceGatheringState === 'complete') {
                        pcs[pi.id].removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                }
                pcs[pi.id].addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(function () {
        var offer = pcs[pi.id].localDescription;
        return fetch(`http://${pi.hostname}:${pi.port}/offer`, {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then(function (response) {
        return response.json();
    }).then(function (answer) {
        return pcs[pi.id].setRemoteDescription(answer);
    }).catch(function (e) {
        alert(e);
    });
}

function start() {
    var config = {
        sdpSemantics: 'unified-plan'
    };

    if (document.getElementById('use-stun').checked) {
        config.iceServers = [{ urls: ['stun:stun.l.google.com:19302'] }];
    }

    pis.map(pi => {
        var pc = new RTCPeerConnection(config);

        // connect audio / video
        pc.addEventListener('track', function (evt) {
            if (evt.track.kind == 'video') {
                document.getElementById(`video_${pi.id}`).srcObject = evt.streams[0];
            } else {
                document.getElementById(`audio_${pi.id}`).srcObject = evt.streams[0];
            }
        });

        pcs[pi.id] = pc;
        negotiate(pi);
    })
   
    document.getElementById('start').style.display = 'none';
    document.getElementById('stop').style.display = 'inline-block';
}

function stop() {
    document.getElementById('stop').style.display = 'none';
    document.getElementById('start').style.display = 'inline-block';
    pis.map(pi => {
        var video_element = document.getElementById(`video_${pi.id}`);
        video_element.pause();
        video_element.removeAttribute('src');
        video_element.load();
    
        // close peer connection
        setTimeout(function () {
            pcs[pi.id].close();
        }, 500);
    })
}
