const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const statusDiv = document.getElementById('status');

let session;
const MODEL_URL = './yolo11n.onnx'; // Step 1で準備したファイル
const INPUT_SIZE = 640; // YOLOのデフォルト入力サイズ

// カメラのセットアップ
async function setupCamera() {
    try {
        // スマホのアウトカメラを優先、Questでは標準カメラ
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment" },
            audio: false
        });
        video.srcObject = stream;
        
        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                // ビデオの解像度に合わせてキャンバスのサイズを設定
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                resolve(video);
            };
        });
    } catch (e) {
        statusDiv.innerText = "カメラのアクセスに失敗しました: " + e.message;
        throw e;
    }
}

// ONNXモデルのロード
async function loadModel() {
    try {
        // WebAssembly (WASM) バックエンドを使用して高速化
        session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'] });
        statusDiv.innerText = "準備完了！";
    } catch (e) {
        statusDiv.innerText = "モデルのロードに失敗: " + e.message;
        throw e;
    }
}

// 映像フレームからYOLO入力用のテンソル（配列）を作成
function prepareInput(videoElement) {
    // 内部用のオフスクリーンキャンバスでリサイズ
    const offscreenCanvas = document.createElement('canvas');
    offscreenCanvas.width = INPUT_SIZE;
    offscreenCanvas.height = INPUT_SIZE;
    const offCtx = offscreenCanvas.getContext('2d');
    
    // 画像を640x640にリサイズして描画
    offCtx.drawImage(videoElement, 0, 0, INPUT_SIZE, INPUT_SIZE);
    const imgData = offCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;

    // RGBA から RGB (Float32Array) に変換し、[1, 3, 640, 640] の形に正規化 (0.0 - 1.0)
    const float32Data = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
        float32Data[i] = imgData[i * 4] / 255.0; // R
        float32Data[i + INPUT_SIZE * INPUT_SIZE] = imgData[i * 4 + 1] / 255.0; // G
        float32Data[i + 2 * INPUT_SIZE * INPUT_SIZE] = imgData[i * 4 + 2] / 255.0; // B
    }

    return new ort.Tensor('float32', float32Data, [1, 3, INPUT_SIZE, INPUT_SIZE]);
}

// 推論と描画のメインループ
async function detectFrame() {
    if (!session) return requestAnimationFrame(detectFrame);

    const inputTensor = prepareInput(video);
    const feeds = { [session.inputNames[0]]: inputTensor }; // モデルの入力名に合わせる

    try {
        // 推論実行
        const results = await session.run(feeds);
        const output = results[session.outputNames[0]]; // 出力テンソル

        // --- ここで出力結果 (output.data) を解析します ---
        // YOLOv11の出力は通常 [1, 84, 8400] のような形状になります。
        // （8400個の候補それぞれに対する x, y, w, h と 80クラスの確率）
        // 実際にはここで NMS (Non-Maximum Suppression) アルゴリズムを実装し、
        // 確率の高いバウンディングボックスを絞り込む必要があります。
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // 【デモ用疑似コード】枠線を描画する処理のイメージ
        // 実際は NMS を経て抽出された配列をループします
        /*
        for (let box of validBoxes) {
            ctx.strokeStyle = '#00FF00';
            ctx.lineWidth = 4;
            ctx.strokeRect(box.x, box.y, box.width, box.height);
            // ラベルの描画
            ctx.fillStyle = '#00FF00';
            ctx.fillText(box.label + ' ' + box.score, box.x, box.y - 5);
        }
        */

    } catch (e) {
        console.error(e);
    }

    // 次のフレームを要求
    requestAnimationFrame(detectFrame);
}

// アプリの起動
async function startApp() {
    await setupCamera();
    await loadModel();
    detectFrame();
}

startApp();
