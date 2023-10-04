def get_paginated_movie_query(offset, limit):
    query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX dbp: <http://dbpedia.org/property/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>

    SELECT ?movie ?title ?director ?releaseDate ?runtime ?wikiPage
    WHERE {{
        ?movie a dbo:Film .
        ?movie foaf:name ?title .
        ?movie dbo:director ?dir .
        ?dir foaf:name ?director .
        ?movie dbo:releaseDate ?releaseDate .
        ?movie dbo:runtime ?runtime .
        ?movie foaf:isPrimaryTopicOf ?wikiPage .
    }}
    OFFSET {offset}
    LIMIT {limit}
    """
    return query


def get_paginated_book_query(offset, limit):
    query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX dbp: <http://dbpedia.org/property/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>

    SELECT ?book ?title ?author ?releaseDate ?pageCount ?wikiPage
    WHERE {{
        ?book a dbo:Book .
        ?book foaf:name ?title .
        ?book dbo:author ?auth .
        ?auth foaf:name ?author .
        ?book dbo:releaseDate ?releaseDate .
        OPTIONAL {{ ?book dbo:numberOfPages ?pageCount . }}
        ?book foaf:isPrimaryTopicOf ?wikiPage .
    }}
    OFFSET {offset}
    LIMIT {limit}
    """
    return query


def get_paginated_song_query(offset, limit):
    query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX dbp: <http://dbpedia.org/property/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>

    SELECT ?song ?title ?artist ?releaseDate ?wikiPage
    WHERE {{
        ?song a dbo:Song .
        ?song foaf:name ?title .
        ?song dbo:artist ?art .
        ?art foaf:name ?artist .
        ?song dbo:releaseDate ?releaseDate .
        ?song foaf:isPrimaryTopicOf ?wikiPage .
    }}
    OFFSET {offset}
    LIMIT {limit}
    """
    return query


def get_paginated_architectural_structure_query(offset, limit):
    query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX dbr: <http://dbpedia.org/resource/>
    PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?landmark ?landmark_name ?type ?country ?latitude ?longitude ?wikipedia_link (IF(bound(?thumbnail), true, false) AS ?has_thumbnail)
    WHERE {{
        ?landmark a dbo:ArchitecturalStructure ;
                dbo:country ?country ;
                geo:lat ?latitude ;
                geo:long ?longitude ;
                foaf:isPrimaryTopicOf ?wikipedia_link ;
                rdf:type ?type ;
                rdfs:label ?landmark_name .

        ?country a dbo:Country .
        ?type rdfs:subClassOf* dbo:ArchitecturalStructure .

        OPTIONAL {{
            ?landmark dbo:thumbnail ?thumbnail .
        }}
        
        FILTER(lang(?landmark_name) = "en")
    }}
    OFFSET {offset}
    LIMIT {limit}
    """
    return query


def get_paginated_populated_place_query(offset, limit):
    query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX dbr: <http://dbpedia.org/resource/>
    PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT ?place ?place_name ?type ?country ?latitude ?longitude ?wikipedia_link ?total_area (xsd:integer(str(?population_total)) AS ?population) (IF(bound(?thumbnail), true, false) AS ?has_thumbnail)
        WHERE {{
            ?place a dbo:PopulatedPlace ;
                dbo:country ?country ;
                geo:lat ?latitude ;
                geo:long ?longitude ;
                foaf:isPrimaryTopicOf ?wikipedia_link ;
                rdf:type ?type ;
                rdfs:label ?place_name .

            ?country a dbo:Country .
            ?type rdfs:subClassOf* dbo:PopulatedPlace .

        OPTIONAL {{
            ?place dbo:thumbnail ?thumbnail .
        }}
        
        OPTIONAL {{
            ?place dbo:areaTotal ?total_area .
        }}

        OPTIONAL {{
            ?place dbo:populationTotal ?population_total .
        }}
        
        FILTER(lang(?place_name) = "en")
    }}
    OFFSET {offset}
    LIMIT {limit}
    """
    return query


def get_paginated_natural_place_query(offset, limit):
    query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX dbr: <http://dbpedia.org/resource/>
    PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?place ?place_name ?type ?country ?latitude ?longitude ?wikipedia_link ?total_area (IF(bound(?thumbnail), true, false) AS ?has_thumbnail)
    WHERE {{
        ?place a dbo:NaturalPlace ;
            dbo:country ?country ;
            geo:lat ?latitude ;
            geo:long ?longitude ;
            foaf:isPrimaryTopicOf ?wikipedia_link ;
            rdf:type ?type ;
            rdfs:label ?place_name .

        ?country a dbo:Country .
        ?type rdfs:subClassOf* dbo:NaturalPlace .

        OPTIONAL {{
            ?place dbo:thumbnail ?thumbnail .
        }}
        
        OPTIONAL {{
            ?place dbo:areaTotal ?total_area .
        }}
        FILTER(lang(?place_name) = "en")
    }}
    OFFSET {offset}
    LIMIT {limit}
    """
    return query
