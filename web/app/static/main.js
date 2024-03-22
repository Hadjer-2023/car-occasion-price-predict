
const brands = ['', 'Ambassador', 'Audi', 'BMW', 'Bentley', 'Chevrolet', 'Datsun', 'Fiat', 'Force',
'Ford', 'Honda', 'Hyundai', 'Isuzu', 'Jaguar', 'Jeep', 'Lamborghini',
'Land', 'Mahindra', 'Maruti', 'Mercedes-Benz', 'Mini', 'Mitsubishi', 'Nissan',
'Porsche', 'Renault', 'Skoda', 'Smart', 'Tata', 'Toyota', 'Volkswagen', 'Volvo']

const locations = ['', 'Ahmedabad', 'Bangalore', 'Chennai', 'Coimbatore', 'Delhi', 'Hyderabad', 'Jaipur', 'Kochi',
'Kolkata', 'Mumbai', 'Pune']

const fuels = {
    "CNG": "CNG",
    "Diesel": "Diesel",
    "Electric": "Électrique",
    "LPG": "GPL",
    "Petrol": "Essence"
}

const transmissions = {
    "Manual": "Manuel",
    "Automatic": "Automatique"
}

const owners = {
    "First": "1ère main",
    "Second": "2ième main",
    "Third": "3ième main",
    "Fourth": "4ième main"
}

const submitCredentials = () => {
    const 
        form = document.getElementById('wrapper_form'),
        loading = document.getElementById('loading')
    
    form.style.display = 'none'
    loading.style.display = 'flex'
}

const convertPrice = price => Math.round(price * 100000 / 87)

const submitPredict = async() => {

    const 
        form = document.getElementById('wrapper_predict_form'),
        loading = document.getElementById('loading'),
        loading_message = document.getElementById('loading_message')
    
    form.style.display = 'none'
    loading.style.display = 'flex'
    loading_message.innerHTML = `Estimation en cours`
    
    const
        brand = document.querySelector('select[name="Name"]').value,
        location = document.querySelector('select[name="Location"]').value,
        year = document.getElementById("Year").value,
        kilometers = document.getElementById("Kilometers_Driven").value,
        fuel = document.querySelector('select[name="Fuel_Type"]').value,
        transmission = document.querySelector('select[name="Transmission"]').value,
        owner_type = document.querySelector('select[name="Owner_Type"]').value,
        engine = document.getElementById("Engine").value,
        power = document.getElementById("Power").value,
        seats = document.querySelector('select[name="Seats"]').value
        data = { brand, location, year, kilometers, fuel, transmission, owner_type, engine, power, seats }

    console.log(data)

    const nb_criterias = Object.values(data).filter(value => value !== '').length

    if (nb_criterias <= 4) {

        form.style.display = 'flex'
        loading.style.display = 'none'
        loading_message.innerHTML = ''

        form_message.textContent = `Veuillez déterminer au moins 5 critères`

        setTimeout(() => form_message.textContent = "", 1111)

    } else {
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
                    let value = data[key]

                    if (value.length > 0) {
                        if (key === 'year') value = `Année ${value}`
                        if (key === 'kilometers') value = `${value} km`
                        if (key === 'fuel') value = `${fuels[value]}`
                        if (key === 'transmission') value = `${transmissions[value]}`
                        if (key === 'owner_type') value = `${owners[value]}`
                        if (key === 'engine') value = `${value} CC`
                        if (key === 'power') value = `${value} BHP`
                        if (key === 'seats') value = `${value} places`
                    }

                    if (value.length)
                        metadata.push(value)      
                }
        
                const formatted = `
                    <p class="formatted_historic">
                        <span>
                            ${metadata.join(' - ')}
                        </span>
                        <span class="estimated_price">
                            ${convertPrice(response.prediction)} €
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

    const brands_select = document.getElementById("brands")

    for (const brand of brands) {
        const option = document.createElement("option")

        option.value = brand
        option.textContent = brand

        brands_select.appendChild(option)
    }

    const locations_select = document.getElementById("locations")

    for (const location of locations) {
        const option = document.createElement("option")

        option.value = location
        option.textContent = location

        locations_select.appendChild(option)
    }

    document.getElementById("button_estimate").addEventListener("click", submitPredict)
})