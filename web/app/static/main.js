
const brands = ['Ambassador', 'Audi', 'BMW', 'Bentley', 'Chevrolet', 'Datsun', 'Fiat', 'Force',
'Ford', 'Honda', 'Hyundai', 'Isuzu', 'Jaguar', 'Jeep', 'Lamborghini',
'Land', 'Mahindra', 'Maruti', 'Mercedes-Benz', 'Mini', 'Mitsubishi', 'Nissan',
'Porsche', 'Renault', 'Skoda', 'Smart', 'Tata', 'Toyota', 'Volkswagen', 'Volvo']

const submitCredentials = () => {
    const 
        form = document.getElementById('wrapper_form'),
        loading = document.getElementById('loading')
    
    form.style.display = 'none'
    loading.style.display = 'flex'
}

const submitPredict = async() => {

    const 
        form = document.getElementById('wrapper_predict_form'),
        loading = document.getElementById('loading'),
        loading_message = document.getElementById('loading_message')
    
    form.style.display = 'none'
    loading.style.display = 'flex'
    loading_message.innerHTML = `
        Prédiction en cours
    `
    
    const
        brand = document.querySelector('select[name="Name"]').value,
        year = document.getElementById("Year").value,
        kilometers = document.getElementById("Kilometers_Driven").value,
        fuel = document.querySelector('select[name="Fuel_Type"]').value,
        transmission = document.querySelector('select[name="Transmission"]').value,
        data = { brand, year, kilometers, fuel, transmission }

    const request = await fetch("http://127.0.0.1:8000/predict", {

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

            const metadata = [] 
        
            for (const key in data) {
                const value = data[key]
    
                if (value.length)
                    metadata.push(value)      
            }
    
            const random_price = Math.floor(Math.random() * (100000 - 1000 + 1)) + 1000
    
            const formatted = `
                <p class="formatted_historic">
                    <span>
                        ${metadata.join(' - ')}
                    </span>
                    <span class="estimated_price">
                        ${random_price} €
                    </span>
                </p>
            `
    
            document.getElementById("content_historic").insertAdjacentHTML('afterbegin', formatted)

        }, 1111)

    } else {
        
        form.style.display = 'flex'
        loading.style.display = 'none'
        loading_message.innerHTML = ''

        form_message.textContent = `Désolé, une erreur avec la requête est survenue, code http : ${response.status}`

        setTimeout(() => form_message.textContent = "", 1111)

    }
}

window.addEventListener('load', () => {

    const queryString = window.location.search,
        params = new URLSearchParams(queryString),
        parameters = {}

    params.forEach((value, key) => { parameters[key] = value })

    console.log(parameters)

    if (parameters.e) {
        const form_message = document.getElementById('form_message')

        let message = ''

        form_message.style.display = 'block'

        if (parameters.e === '0') {
            message = 'Une erreur est survenue'    
        } else if (parameters.e === '1') {
            message = 'Identifiants incorrects'    
        } else if (parameters.e === '2') {
            message = 'Les mots de passe sont différents'    
        } else if (parameters.e === '3') {
            message = 'Les champs ne sont pas bien remplis'    
        }

        form_message.innerText = message

        setTimeout(() => form_message.style.display = 'none', 1500)
    }

    const brands = ['Ambassador', 'Audi', 'BMW', 'Bentley', 'Chevrolet', 'Datsun', 'Fiat', 'Force',
    'Ford', 'Honda', 'Hyundai', 'Isuzu', 'Jaguar', 'Jeep', 'Lamborghini',
    'Land', 'Mahindra', 'Maruti', 'Mercedes-Benz', 'Mini', 'Mitsubishi', 'Nissan',
    'Porsche', 'Renault', 'Skoda', 'Smart', 'Tata', 'Toyota', 'Volkswagen', 'Volvo']

    const brands_select = document.getElementById("brands")

    for (const brand of brands) {
        const option = document.createElement("option")

        option.value = brand
        option.textContent = brand

        brands_select.appendChild(option)
    }

    document.getElementById("button_estimate").addEventListener("click", submitPredict)
})