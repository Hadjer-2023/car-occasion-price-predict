
window.addEventListener('load', () => {

    const models = {
        gbr: { name: 'Gradient Boosting Regressor' },
        lr: { name: 'Linear Regression' }
    }

    const button_training = document.getElementById('button_training')

    if (button_training)
        button_training.addEventListener('click', async(event) => {
    
            event.preventDefault()

            const 
                form = document.getElementById('form_train'),
                loading = document.getElementById('loading'),
                loading_message = document.getElementById('loading_message'),
                model = document.querySelector('select[name="Model"]').value
            
            form.style.display = 'none'
            loading.style.display = 'flex'
            loading_message.innerHTML = `
                Entraînement du modèle <span style="font-weight: bold;">${models[model].name}</span> en cours
            `

            const 
                access_token = localStorage.getItem('access_token'),
                data = {
                    access_token,
                    model,
                    hyper_params: { one: 'one', two: 'two' }
                }
        
            const request = await fetch("/train", {

                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(data)
                
            })
        
            const response = await request.json()
            console.log(response)

            if (request.ok) {

                setTimeout(() => {
    
                    form.style.display = 'flex'
                    loading.style.display = 'none'
                    loading_message.innerHTML = ''

                    if (!response.fail) {

                        const formatted = `
                                <div class="formatted_historic">
                                    <h4>Entraînement avec ${models[model].name}</h4>
                                    <p>R2: ${response.r2}</p>
                                    <p>MAE: ${response.mae}</p>
                                    <p>MSE: ${response.mse}</p>
                                    <p>RMSE: ${response.rmse}</p>
                                </div>
                        `
        
                        document.getElementById("content_historic").insertAdjacentHTML('afterbegin', formatted)
    
                    } else {
    
                        form_message.insertAdjacentHTML('afterbegin', `${response.fail} <a href="/">Reconnexion</a>`)

                    }
    
                }, 1111)

            } else {
                
                form.style.display = 'flex'
                loading.style.display = 'none'
                loading_message.innerHTML = ''

                form_message.textContent = `Désolé, une erreur avec la requête est survenue, code http : ${response.status}`

                setTimeout(() => form_message.textContent = "", 1111)
        
            }
        })

})


