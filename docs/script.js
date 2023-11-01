class CanvasDrawer {
    constructor(canvasId, convertButtonId) {
        this.canvas = document.getElementById(canvasId)
        this.ctx = this.canvas.getContext('2d')
        this.convertButton = document.getElementById(convertButtonId)
        this.ctx.fillStyle = "black";
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.drawing = false
        this.mousePos = { x: 0, y: 0 }
        this.lastPos = this.mousePos

        this.rect = this.canvas.getBoundingClientRect()
        
        this.addEventListeners()
        this.drawLoop()

    }

    getMousePos(mouseEvent) {
        return {
            x: mouseEvent.clientX - this.rect.left,
            y: mouseEvent.clientY - this.rect.top
        }
    }

    renderCanvas() {
        // If the mouse is down
        if (this.drawing) {
            this.ctx.beginPath() // Start the path 
            this.ctx.strokeStyle = "white";
            this.ctx.lineWidth = 10 // Set the line width to 10 so the stroke aren't erased when rescaling to 28x28px
            this.ctx.moveTo(this.lastPos.x, this.lastPos.y) // Move the start of the line to the position the mouse ended on last render
            this.ctx.lineTo(this.mousePos.x, this.mousePos.y) // Draw a line to the current position of the mouse
            this.ctx.stroke() // Draw the new segment to the canva (apply current changes)
            this.ctx.closePath() // Close the path of the new segment
            this.lastPos = this.mousePos
        }
    }
    
    drawLoop() {
        requestAnimationFrame(() => this.drawLoop())
        this.renderCanvas()
    }

    addEventListeners() {
        this.canvas.addEventListener("mousedown", (e) => {
            this.drawing = true
            this.lastPos = this.getMousePos(e)
        })

        this.canvas.addEventListener("mouseup", () => {
            this.drawing = false
        })

        this.canvas.addEventListener("mousemove", (e) => {
            this.mousePos = this.getMousePos(e)
        })
        
        this.canvas.addEventListener("mouseout", () => {
            this.drawing = false
        })
    
        this.canvas.addEventListener("mouseenter", (e) => {
            if (e.buttons === 1) { // Check if left button is pressed
                this.drawing = true
                this.lastPos = this.getMousePos(e)
            }
        })
        document.getElementById('clear').addEventListener('click', () => {
            this.ctx.reset()
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        })
    }


    getResizedGrayscaleArray() {
        // Create a temporary canvas to resize the image
        let tempCanvas = document.createElement('canvas')
        let tempCtx = tempCanvas.getContext('2d')
        const resizeWidth = 28
        const resizeHeight = 28
        tempCanvas.width = resizeWidth
        tempCanvas.height = resizeHeight

        // Draw the original canvas onto the temp canvas, resized
        tempCtx.drawImage(this.canvas, 0, 0, this.canvas.width, this.canvas.height, 0, 0, resizeWidth, resizeHeight)

        // Extract the pixel data from the resized canvas
        const resizedData = tempCtx.getImageData(0, 0, resizeWidth, resizeHeight).data
        return new Float32Array(resizedData.filter((e, i) => i%4==0))
    }

}


class ONNXModelHandler {
    constructor(modelPath) {
        this.modelPath = modelPath
        this.session = new onnx.InferenceSession()
    }
    
    async loadModel() {
        try {
            await this.session.loadModel(this.modelPath)
            console.log("Model loaded successfully.")
        } catch (error) {
            console.error("Failed to load model:", error)
        }
    }
    
    async runInference(inputTensor) {
        try {
            const outputMap = await this.session.run([inputTensor])
            return outputMap
        } catch (error) {
            console.error("Error during inference:", error)
            return null
        }
    }

    softmax(arr) {
        const maxLogit = Math.max(...arr);
        const sum = arr.reduce((acc, logit) => acc + Math.exp(logit - maxLogit), 0);
        return arr.map(logit => Math.exp(logit - maxLogit) / sum);
    }
    
    
}

const canvasDrawer = new CanvasDrawer('canvas', 'convert')
const modelHandler = new ONNXModelHandler('./fashionMNIST.onnx')
modelHandler.loadModel().then(() => {
    setInterval(() => {
        const canvasData = canvasDrawer.getResizedGrayscaleArray()
        const tensor = new onnx.Tensor(canvasData, 'float32', [1, 1, 28, 28])
        modelHandler.runInference(tensor).then((outputMap) => {
            let logits = outputMap.values().next().value.data;
            let probabilities = modelHandler.softmax(logits);
            probabilities.forEach((prob, index) => {
                document.getElementById(`label-${index}`).innerText = `${(prob * 100).toFixed(2)}%`;
            });
        })
    })
}, 500)

