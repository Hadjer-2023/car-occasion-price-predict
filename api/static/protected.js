
const DECIMALS = 4

window.addEventListener('load', () => {

    let 
        BEST_CROSS_R2 = 0,
        BEST_PRED_R2 = 0,
        BEST_CROSS_MODEL = '',
        BEST_PRED_MODEL = '',
        BEST_HYPARAMS = ''

    const models = {
        lr: { name: 'Linear Regression', hyparams: {} },
        l: { name: 'Lasso', hyparams: { "alpha": [0.1, 1.0, 10.0] } },
        r: { name: 'Ridge', hyparams: { "alpha": [0.1, 1.0, 10.0] } },
        en: { name: 'ElasticNet', hyparams: { "alpha": [0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 0.9] } },
        abr: { name: 'Ada Boost Regressor', hyparams:{ "n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1.0] } },
        gbr: { name: 'Gradient Boosting Regressor', hyparams: { "n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1.0] } },
        rfr: { name: 'Random Forest Regressor', hyparams: { "max_depth": [3, 5, 10], "n_estimators": [10, 50, 100] } }
    }

    const processes = {
        cv: { name: 'Cross val score' },
        gs: { name: 'Grid search cv' }
    }

    const 
        select_estimator = document.getElementById('select_estimator'),
        select_process = document.getElementById('select_process'),
        div_hyparams = document.getElementById('hyparams')

    if (select_process)
        select_process.addEventListener('change', event => {

            if (select_process.value === 'gs') {
                const 
                    estimator = select_estimator.value,
                    hyparams = models[estimator].hyparams

                div_hyparams.innerHTML = ''

                if (Object.keys(hyparams).length > 0) {
                    let html_hyparams = '<div class="info_hyparams"><p>&#8505;</p> séparé par <p>,</p> pour plusieurs valeurs</div>'

                    for (const hyparam in hyparams) {
                        const default_values = hyparams[hyparam]
                        html_hyparams += `
                            <div class="div_hyparams">
                                <label for="${hyparam}">
                                    ${hyparam}
                                </label>
                                <input type="text" data-hyparams name="${hyparam}" value="${default_values.join(',')}"/>
                            </div>
                        `
                    }
    
                    div_hyparams.insertAdjacentHTML('afterbegin', html_hyparams)
                }

            } else {

                div_hyparams.innerHTML = ''

            }   
            
        })

    if (select_estimator)
        select_estimator.addEventListener('change', event => {
            if (select_process.value === 'gs') {
                const 
                    estimator = select_estimator.value,
                    hyparams = models[estimator].hyparams

                div_hyparams.innerHTML = ''

                if (Object.keys(hyparams).length > 0) {
                    let html_hyparams = '<div class="info_hyparams"><p>&#8505;</p> séparé par <p>,</p> pour plusieurs valeurs</div>'

                    for (const hyparam in hyparams) {
                        const default_values = hyparams[hyparam]
                        html_hyparams += `
                            <div class="div_hyparams">
                                <label for="${hyparam}">
                                    ${hyparam}
                                </label>
                                <input type="text" data-hyparams name="${hyparam}" value="${default_values.join(',')}"/>
                            </div>
                        `
                    }
    
                    div_hyparams.insertAdjacentHTML('afterbegin', html_hyparams)
                }

            } else {

                div_hyparams.innerHTML = ''

            }   
        
    })
    
    const button_training = document.getElementById('button_training')

    if (button_training)
        button_training.addEventListener('click', async(event) => {
    
            event.preventDefault()

            const 
                form = document.getElementById('form_train'),
                loading = document.getElementById('loading'),
                best_scores = document.getElementById('best_scores'),
                loading_message = document.getElementById('loading_message'),
                model = document.querySelector('select[name="Model"]').value,
                process = document.querySelector('select[name="Process"]').value
                hyparams = {}

            let best_cross_r2 = best_pred_r2 = ''
            
            form.style.display = 'none'
            loading.style.display = 'flex'

            if (process === 'gs') {
                const input_hyparams = document.querySelectorAll("input[data-hyparams]")
                
                for (const hyparam of input_hyparams)
                    hyparams[hyparam.name] = hyparam.value.split(',').map(x => parseFloat(x))
                
            }

            loading_message.innerHTML = `
                Entraînement avec l'estimateur <span style="font-weight: bold;">${models[model].name}</span>
                en mode <span style="font-weight: bold;">${processes[process].name}</span> en cours...
                <br />
                <br />
                ${Object.keys(hyparams).length > 0 ? `<p>Hyper paramatètres</p><pre>${JSON.stringify(hyparams)}</pre>` : ''} 
            `

            const 
                access_token = localStorage.getItem('access_token'),
                data = {
                    access_token,
                    model,
                    process,
                    hyparams
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

                        let formatted = `
                                <div class="formatted_historic">
                                    <h4>${models[model].name} (${processes[process].name})</h4>
                                    ${Object.keys(hyparams).length > 0 ? `<p class="sended_hyparams">${JSON.stringify(hyparams)}</p>` : ''} 
                                    <p>Temps d'exécution : ${response.execution_time.toFixed(2)}s</p>
                                    <p>MAE : ${response.mae.toFixed(2)}</p>
                                    <p>MSE : ${response.mse.toFixed(2)}</p>
                                    <p>RMSE : ${response.rmse.toFixed(2)}</p>
                        `

                        if (process === 'cv') {

                            r2_cross_mean = response.r2_cross_mean.toFixed(DECIMALS)
                            r2_cross_pred = response.r2_cross_pred.toFixed(DECIMALS)

                            if (r2_cross_mean > BEST_CROSS_R2) {
                                BEST_CROSS_R2 = r2_cross_mean
                                BEST_CROSS_MODEL = models[model].name

                                best_cross_r2 = `
                                    <p>Best R2 cross: ${r2_cross_mean}</p>
                                    <p>${models[model].name} (Cross val score)</p>
                                `
                            } else {

                                best_cross_r2 = `
                                    <p>Best R2 cross: ${BEST_CROSS_R2}</p>
                                    <p>${BEST_CROSS_MODEL} (Cross val score)</p>
                                `

                            }

                            if (r2_cross_pred > BEST_PRED_R2) {
                                BEST_PRED_R2 = r2_cross_pred
                                BEST_PRED_MODEL = models[model].name

                                best_pred_r2 = `
                                    <p>Best R2 pred: ${r2_cross_pred}</p>
                                    <p>${models[model].name} (Cross val score)</p>
                                `
                            } else {

                                best_pred_r2 = `
                                    <p>Best R2 pred: ${BEST_PRED_R2}</p>
                                    <p>${BEST_PRED_MODEL} (GCross val score)</p>
                                `

                            }

                            formatted += `
                                <p>R2 cross : ${r2_cross_mean}</p>
                                <p>R2 pred : ${r2_cross_pred}</p>
                            `

                        } else {

                            const 
                                best_params = response.best_params,
                                length = Object.keys(best_params).length

                            best_score = response.best_score.toFixed(DECIMALS)
                            r2_grid = response.r2_grid.toFixed(DECIMALS)

                            if (best_score > BEST_CROSS_R2) {
                                BEST_CROSS_R2 = best_score
                                BEST_CROSS_MODEL = models[model].name
                                BEST_HYPARAMS = length > 0 ? `<p><small>${JSON.stringify(best_params)}</small></p>` : ''

                                best_cross_r2 = `
                                    <p>Best R2 cross: ${best_score}</p>
                                    <p>${models[model].name} (Grid search cv)</p>
                                    ${length > 0 ? `<p><small>${JSON.stringify(best_params)}</small></p>` : ''}
                                `
                            } else {

                                best_cross_r2 = `
                                    <p>Best R2 cross: ${BEST_CROSS_R2}</p>
                                    <p>${BEST_CROSS_MODEL} (Grid search cv)</p>
                                    ${BEST_HYPARAMS}
                                `

                            }

                            if (r2_grid > BEST_PRED_R2) {
                                BEST_PRED_R2 = r2_grid
                                BEST_PRED_MODEL = models[model].name
                                BEST_HYPARAMS = length > 0 ? `<p><small>${JSON.stringify(best_params)}</small></p>` : ''

                                best_pred_r2 = `
                                    <p>Best R2 pred: ${r2_grid}</p>
                                    <p>${models[model].name} (Grid search cv)</p>
                                    ${length > 0 ? `<p><small>${JSON.stringify(best_params)}</small></p>` : ''}
                                `
                            } else {

                                best_pred_r2 = `
                                    <p>Best R2 pred: ${BEST_PRED_R2}</p>
                                    <p>${BEST_PRED_MODEL} (Grid search cv)</p>
                                    ${BEST_HYPARAMS}
                                `

                            }

                            formatted += `
                                <p>R2 cross : ${best_score}</p>
                                <p>R2 pred : ${r2_grid}</p>
                            `

                            if (length > 0) {
                                formatted += length > 1 ? `<p>Meilleurs hyperparamètres :</p>` : `<p>Meilleur hyperparamètre :</p>`
                                for (const key in best_params) 
                                    formatted += `<p class="hyparams"> - ${key}: ${best_params[key]}</p>`
                            }
                        }

                        formatted += '</div>'
        
                        document.getElementById("content_historic").insertAdjacentHTML('afterbegin', formatted)

                        if (best_scores.style.display = 'flex')

                        best_scores.innerHTML = `
                            <h4>Meilleurs scores</h4>
                            <div>${best_cross_r2}</div>
                            <hr />
                            <div>${best_pred_r2}</div>
                        `
    
                    } else {
    
    
                        form_message.insertAdjacentHTML('afterbegin', `${response.message}`)

                    }
    
                }, 1111)

            } else {
                
                form.style.display = 'flex'
                loading.style.display = 'none'
                loading_message.innerHTML = ''

                if (response.detail) {

                    form_message.insertAdjacentHTML('afterbegin', `Jeton expiré : <a href="/">Authentification</a>`)

                } else {

                    form_message.textContent = `Désolé, une erreur lors la requête est survenue`

                    setTimeout(() => form_message.textContent = "", 1111)

                }
        
            }
        })

})


