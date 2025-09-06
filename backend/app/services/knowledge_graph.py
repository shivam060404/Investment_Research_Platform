from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
import re
from loguru import logger

# Neo4j imports
try:
    from neo4j import GraphDatabase, Driver
    from neo4j.exceptions import ServiceUnavailable, AuthError
except ImportError:
    GraphDatabase = None
    Driver = None
    logger.warning("Neo4j driver not available, using mock implementation")

from ..core.config import settings
from ..core.database import Neo4jConnection

class EntityType(Enum):
    """Types of entities in the knowledge graph"""
    COMPANY = "Company"
    PERSON = "Person"
    SECTOR = "Sector"
    INDUSTRY = "Industry"
    FINANCIAL_METRIC = "FinancialMetric"
    EVENT = "Event"
    PRODUCT = "Product"
    LOCATION = "Location"
    REGULATION = "Regulation"
    MARKET_INDEX = "MarketIndex"

class RelationshipType(Enum):
    """Types of relationships in the knowledge graph"""
    OWNS = "OWNS"
    COMPETES_WITH = "COMPETES_WITH"
    SUPPLIES_TO = "SUPPLIES_TO"
    ACQUIRED_BY = "ACQUIRED_BY"
    PART_OF = "PART_OF"
    LOCATED_IN = "LOCATED_IN"
    REPORTS = "REPORTS"
    INFLUENCES = "INFLUENCES"
    CORRELATES_WITH = "CORRELATES_WITH"
    MANAGES = "MANAGES"
    INVESTS_IN = "INVESTS_IN"
    REGULATED_BY = "REGULATED_BY"

@dataclass
class Entity:
    """Represents an entity in the knowledge graph"""
    id: str
    name: str
    entity_type: EntityType
    properties: Dict[str, Any]
    created_at: datetime = None
    updated_at: datetime = None

@dataclass
class Relationship:
    """Represents a relationship between entities"""
    from_entity_id: str
    to_entity_id: str
    relationship_type: RelationshipType
    properties: Dict[str, Any]
    strength: float = 1.0
    created_at: datetime = None
    updated_at: datetime = None

class EntityExtractor:
    """Extracts entities from financial text and data"""
    
    def __init__(self):
        self.company_patterns = [
            r'\b([A-Z][a-z]+ (?:Inc|Corp|Corporation|Ltd|Limited|LLC|Co|Company))\b',
            r'\b([A-Z]{2,5})\s+(?:stock|shares|equity)\b',
            r'\b([A-Z][a-zA-Z\s&]+)\s+(?:reported|announced|disclosed)\b'
        ]
        
        self.person_patterns = [
            r'\b(?:CEO|CFO|CTO|President|Chairman|Director)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b',
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+),\s+(?:CEO|CFO|CTO|President|Chairman)\b'
        ]
        
        self.financial_metric_patterns = [
            r'\b(revenue|profit|earnings|EBITDA|cash flow|debt|assets|liabilities)\b',
            r'\b(P/E ratio|ROE|ROA|gross margin|net margin|debt-to-equity)\b'
        ]
        
        self.known_companies = {
            "AAPL": "Apple Inc.",
            "GOOGL": "Alphabet Inc.",
            "MSFT": "Microsoft Corporation",
            "TSLA": "Tesla Inc.",
            "AMZN": "Amazon.com Inc.",
            "META": "Meta Platforms Inc.",
            "NVDA": "NVIDIA Corporation",
            "JPM": "JPMorgan Chase & Co.",
            "JNJ": "Johnson & Johnson",
            "V": "Visa Inc."
        }
        
        self.sectors = {
            "Technology", "Healthcare", "Financial Services", "Consumer Discretionary",
            "Communication Services", "Industrials", "Consumer Staples", "Energy",
            "Utilities", "Real Estate", "Materials"
        }
    
    def extract_entities(self, text: str, context: Dict[str, Any] = None) -> List[Entity]:
        """Extract entities from text"""
        entities = []
        
        # Extract companies
        entities.extend(self._extract_companies(text, context))
        
        # Extract people
        entities.extend(self._extract_people(text, context))
        
        # Extract financial metrics
        entities.extend(self._extract_financial_metrics(text, context))
        
        # Extract sectors/industries
        entities.extend(self._extract_sectors(text, context))
        
        return entities
    
    def _extract_companies(self, text: str, context: Dict[str, Any] = None) -> List[Entity]:
        """Extract company entities"""
        companies = []
        
        # Check for known ticker symbols
        for ticker, company_name in self.known_companies.items():
            if ticker in text or company_name in text:
                entity = Entity(
                    id=f"company_{ticker.lower()}",
                    name=company_name,
                    entity_type=EntityType.COMPANY,
                    properties={
                        "ticker": ticker,
                        "full_name": company_name,
                        "source": "known_companies"
                    },
                    created_at=datetime.now()
                )
                companies.append(entity)
        
        # Extract using patterns
        for pattern in self.company_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                company_name = match.group(1)
                entity = Entity(
                    id=f"company_{company_name.lower().replace(' ', '_')}",
                    name=company_name,
                    entity_type=EntityType.COMPANY,
                    properties={
                        "full_name": company_name,
                        "source": "pattern_extraction",
                        "confidence": 0.8
                    },
                    created_at=datetime.now()
                )
                companies.append(entity)
        
        return companies
    
    def _extract_people(self, text: str, context: Dict[str, Any] = None) -> List[Entity]:
        """Extract person entities"""
        people = []
        
        for pattern in self.person_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                person_name = match.group(1)
                entity = Entity(
                    id=f"person_{person_name.lower().replace(' ', '_')}",
                    name=person_name,
                    entity_type=EntityType.PERSON,
                    properties={
                        "full_name": person_name,
                        "source": "pattern_extraction",
                        "context": match.group(0)
                    },
                    created_at=datetime.now()
                )
                people.append(entity)
        
        return people
    
    def _extract_financial_metrics(self, text: str, context: Dict[str, Any] = None) -> List[Entity]:
        """Extract financial metric entities"""
        metrics = []
        
        for pattern in self.financial_metric_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                metric_name = match.group(1)
                entity = Entity(
                    id=f"metric_{metric_name.lower().replace(' ', '_').replace('/', '_')}",
                    name=metric_name,
                    entity_type=EntityType.FINANCIAL_METRIC,
                    properties={
                        "metric_name": metric_name,
                        "source": "pattern_extraction"
                    },
                    created_at=datetime.now()
                )
                metrics.append(entity)
        
        return metrics
    
    def _extract_sectors(self, text: str, context: Dict[str, Any] = None) -> List[Entity]:
        """Extract sector/industry entities"""
        sectors = []
        
        for sector in self.sectors:
            if sector.lower() in text.lower():
                entity = Entity(
                    id=f"sector_{sector.lower().replace(' ', '_')}",
                    name=sector,
                    entity_type=EntityType.SECTOR,
                    properties={
                        "sector_name": sector,
                        "source": "known_sectors"
                    },
                    created_at=datetime.now()
                )
                sectors.append(entity)
        
        return sectors

class RelationshipExtractor:
    """Extracts relationships between entities"""
    
    def __init__(self):
        self.relationship_patterns = {
            RelationshipType.OWNS: [
                r'(\w+)\s+owns\s+(\w+)',
                r'(\w+)\s+acquired\s+(\w+)',
                r'(\w+)\s+subsidiary\s+(\w+)'
            ],
            RelationshipType.COMPETES_WITH: [
                r'(\w+)\s+competes?\s+with\s+(\w+)',
                r'(\w+)\s+rival\s+(\w+)',
                r'(\w+)\s+vs\.?\s+(\w+)'
            ],
            RelationshipType.SUPPLIES_TO: [
                r'(\w+)\s+supplies\s+to\s+(\w+)',
                r'(\w+)\s+vendor\s+for\s+(\w+)',
                r'(\w+)\s+provides\s+.+\s+to\s+(\w+)'
            ],
            RelationshipType.PART_OF: [
                r'(\w+)\s+is\s+part\s+of\s+(\w+)',
                r'(\w+)\s+division\s+of\s+(\w+)',
                r'(\w+)\s+unit\s+of\s+(\w+)'
            ]
        }
    
    def extract_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships from text given known entities"""
        relationships = []
        entity_map = {entity.name.lower(): entity for entity in entities}
        
        # Extract using patterns
        for rel_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity1_name = match.group(1).lower()
                    entity2_name = match.group(2).lower()
                    
                    entity1 = entity_map.get(entity1_name)
                    entity2 = entity_map.get(entity2_name)
                    
                    if entity1 and entity2:
                        relationship = Relationship(
                            from_entity_id=entity1.id,
                            to_entity_id=entity2.id,
                            relationship_type=rel_type,
                            properties={
                                "source": "pattern_extraction",
                                "context": match.group(0),
                                "confidence": 0.7
                            },
                            created_at=datetime.now()
                        )
                        relationships.append(relationship)
        
        # Extract implicit relationships
        relationships.extend(self._extract_implicit_relationships(entities))
        
        return relationships
    
    def _extract_implicit_relationships(self, entities: List[Entity]) -> List[Relationship]:
        """Extract implicit relationships based on entity types and properties"""
        relationships = []
        
        companies = [e for e in entities if e.entity_type == EntityType.COMPANY]
        sectors = [e for e in entities if e.entity_type == EntityType.SECTOR]
        people = [e for e in entities if e.entity_type == EntityType.PERSON]
        metrics = [e for e in entities if e.entity_type == EntityType.FINANCIAL_METRIC]
        
        # Companies in same sector compete
        for sector in sectors:
            sector_companies = [c for c in companies if sector.name.lower() in str(c.properties).lower()]
            for i, company1 in enumerate(sector_companies):
                for company2 in sector_companies[i+1:]:
                    relationship = Relationship(
                        from_entity_id=company1.id,
                        to_entity_id=company2.id,
                        relationship_type=RelationshipType.COMPETES_WITH,
                        properties={
                            "source": "implicit_sector_competition",
                            "sector": sector.name,
                            "confidence": 0.5
                        },
                        strength=0.5,
                        created_at=datetime.now()
                    )
                    relationships.append(relationship)
        
        # Companies report financial metrics
        for company in companies:
            for metric in metrics:
                relationship = Relationship(
                    from_entity_id=company.id,
                    to_entity_id=metric.id,
                    relationship_type=RelationshipType.REPORTS,
                    properties={
                        "source": "implicit_financial_reporting",
                        "confidence": 0.6
                    },
                    strength=0.6,
                    created_at=datetime.now()
                )
                relationships.append(relationship)
        
        return relationships

class Neo4jKnowledgeGraph:
    """Neo4j-based knowledge graph implementation"""
    
    def __init__(self):
        self.driver = None
        self.is_connected = False
        self.entity_extractor = EntityExtractor()
        self.relationship_extractor = RelationshipExtractor()
    
    async def initialize(self):
        """Initialize Neo4j connection"""
        try:
            if GraphDatabase:
                self.driver = GraphDatabase.driver(
                    settings.NEO4J_URI,
                    auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
                )
                
                # Test connection
                with self.driver.session() as session:
                    result = session.run("RETURN 1 as test")
                    test_value = result.single()["test"]
                    if test_value == 1:
                        self.is_connected = True
                        logger.info("Connected to Neo4j successfully")
                
                # Create constraints and indexes
                await self._create_schema()
            else:
                logger.warning("Neo4j driver not available, using mock implementation")
                self.is_connected = True
                
        except Exception as e:
            logger.error(f"Error connecting to Neo4j: {e}")
            self.is_connected = False
    
    async def _create_schema(self):
        """Create Neo4j schema constraints and indexes"""
        try:
            with self.driver.session() as session:
                # Create uniqueness constraints
                constraints = [
                    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
                    "CREATE CONSTRAINT company_id IF NOT EXISTS FOR (c:Company) REQUIRE c.id IS UNIQUE",
                    "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE"
                ]
                
                for constraint in constraints:
                    try:
                        session.run(constraint)
                    except Exception as e:
                        logger.debug(f"Constraint may already exist: {e}")
                
                # Create indexes
                indexes = [
                    "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)"
                ]
                
                for index in indexes:
                    try:
                        session.run(index)
                    except Exception as e:
                        logger.debug(f"Index may already exist: {e}")
                
                logger.info("Neo4j schema created successfully")
                
        except Exception as e:
            logger.error(f"Error creating Neo4j schema: {e}")
    
    async def add_entity(self, entity: Entity) -> bool:
        """Add entity to knowledge graph"""
        if not self.is_connected:
            logger.warning("Not connected to Neo4j, using mock implementation")
            return True
        
        try:
            with self.driver.session() as session:
                query = """
                MERGE (e:Entity {id: $id})
                SET e.name = $name,
                    e.entity_type = $entity_type,
                    e.properties = $properties,
                    e.created_at = $created_at,
                    e.updated_at = $updated_at
                WITH e
                CALL apoc.create.addLabels(e, [$entity_type]) YIELD node
                RETURN e
                """
                
                result = session.run(query, {
                    "id": entity.id,
                    "name": entity.name,
                    "entity_type": entity.entity_type.value,
                    "properties": json.dumps(entity.properties),
                    "created_at": entity.created_at.isoformat() if entity.created_at else datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                })
                
                return result.single() is not None
                
        except Exception as e:
            logger.error(f"Error adding entity to Neo4j: {e}")
            return False
    
    async def add_relationship(self, relationship: Relationship) -> bool:
        """Add relationship to knowledge graph"""
        if not self.is_connected:
            logger.warning("Not connected to Neo4j, using mock implementation")
            return True
        
        try:
            with self.driver.session() as session:
                query = f"""
                MATCH (from:Entity {{id: $from_id}})
                MATCH (to:Entity {{id: $to_id}})
                MERGE (from)-[r:{relationship.relationship_type.value}]->(to)
                SET r.properties = $properties,
                    r.strength = $strength,
                    r.created_at = $created_at,
                    r.updated_at = $updated_at
                RETURN r
                """
                
                result = session.run(query, {
                    "from_id": relationship.from_entity_id,
                    "to_id": relationship.to_entity_id,
                    "properties": json.dumps(relationship.properties),
                    "strength": relationship.strength,
                    "created_at": relationship.created_at.isoformat() if relationship.created_at else datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                })
                
                return result.single() is not None
                
        except Exception as e:
            logger.error(f"Error adding relationship to Neo4j: {e}")
            return False
    
    async def find_entity(self, entity_id: str) -> Optional[Entity]:
        """Find entity by ID"""
        if not self.is_connected:
            return None
        
        try:
            with self.driver.session() as session:
                query = "MATCH (e:Entity {id: $id}) RETURN e"
                result = session.run(query, {"id": entity_id})
                record = result.single()
                
                if record:
                    node = record["e"]
                    return Entity(
                        id=node["id"],
                        name=node["name"],
                        entity_type=EntityType(node["entity_type"]),
                        properties=json.loads(node.get("properties", "{}")),
                        created_at=datetime.fromisoformat(node.get("created_at", datetime.now().isoformat())),
                        updated_at=datetime.fromisoformat(node.get("updated_at", datetime.now().isoformat()))
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"Error finding entity in Neo4j: {e}")
            return None
    
    async def find_related_entities(self, entity_id: str, relationship_types: List[RelationshipType] = None, 
                                  max_depth: int = 2) -> List[Tuple[Entity, List[Relationship]]]:
        """Find entities related to given entity"""
        if not self.is_connected:
            return []
        
        try:
            with self.driver.session() as session:
                # Build relationship type filter
                rel_filter = ""
                if relationship_types:
                    rel_types = "|".join([rt.value for rt in relationship_types])
                    rel_filter = f":{rel_types}"
                
                query = f"""
                MATCH path = (start:Entity {{id: $entity_id}})-[r{rel_filter}*1..{max_depth}]-(related:Entity)
                RETURN related, relationships(path) as rels
                LIMIT 50
                """
                
                result = session.run(query, {"entity_id": entity_id})
                
                related_entities = []
                for record in result:
                    related_node = record["related"]
                    relationships = record["rels"]
                    
                    entity = Entity(
                        id=related_node["id"],
                        name=related_node["name"],
                        entity_type=EntityType(related_node["entity_type"]),
                        properties=json.loads(related_node.get("properties", "{}")),
                        created_at=datetime.fromisoformat(related_node.get("created_at", datetime.now().isoformat()))
                    )
                    
                    # Convert Neo4j relationships to our Relationship objects
                    rel_objects = []
                    for rel in relationships:
                        rel_obj = Relationship(
                            from_entity_id=rel.start_node["id"],
                            to_entity_id=rel.end_node["id"],
                            relationship_type=RelationshipType(rel.type),
                            properties=json.loads(rel.get("properties", "{}")),
                            strength=rel.get("strength", 1.0)
                        )
                        rel_objects.append(rel_obj)
                    
                    related_entities.append((entity, rel_objects))
                
                return related_entities
                
        except Exception as e:
            logger.error(f"Error finding related entities in Neo4j: {e}")
            return []
    
    async def search_entities(self, query: str, entity_types: List[EntityType] = None, 
                            limit: int = 20) -> List[Entity]:
        """Search entities by name or properties"""
        if not self.is_connected:
            return []
        
        try:
            with self.driver.session() as session:
                # Build entity type filter
                type_filter = ""
                if entity_types:
                    types = "|".join([et.value for et in entity_types])
                    type_filter = f" AND e.entity_type IN [{', '.join([f"'{t}'" for t in types.split('|')])}]"
                
                cypher_query = f"""
                MATCH (e:Entity)
                WHERE e.name CONTAINS $query{type_filter}
                RETURN e
                ORDER BY e.name
                LIMIT {limit}
                """
                
                result = session.run(cypher_query, {"query": query})
                
                entities = []
                for record in result:
                    node = record["e"]
                    entity = Entity(
                        id=node["id"],
                        name=node["name"],
                        entity_type=EntityType(node["entity_type"]),
                        properties=json.loads(node.get("properties", "{}")),
                        created_at=datetime.fromisoformat(node.get("created_at", datetime.now().isoformat()))
                    )
                    entities.append(entity)
                
                return entities
                
        except Exception as e:
            logger.error(f"Error searching entities in Neo4j: {e}")
            return []
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        if not self.is_connected:
            return {
                "total_entities": 0,
                "total_relationships": 0,
                "entity_types": {},
                "relationship_types": {},
                "connected": False
            }
        
        try:
            with self.driver.session() as session:
                # Count entities by type
                entity_query = """
                MATCH (e:Entity)
                RETURN e.entity_type as type, count(e) as count
                """
                entity_result = session.run(entity_query)
                entity_types = {record["type"]: record["count"] for record in entity_result}
                
                # Count relationships by type
                rel_query = """
                MATCH ()-[r]-()
                RETURN type(r) as type, count(r)/2 as count
                """
                rel_result = session.run(rel_query)
                relationship_types = {record["type"]: record["count"] for record in rel_result}
                
                # Total counts
                total_entities = sum(entity_types.values())
                total_relationships = sum(relationship_types.values())
                
                return {
                    "total_entities": total_entities,
                    "total_relationships": total_relationships,
                    "entity_types": entity_types,
                    "relationship_types": relationship_types,
                    "connected": True,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting graph statistics: {e}")
            return {"error": str(e), "connected": False}
    
    async def process_document(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process document to extract and store entities and relationships"""
        try:
            # Extract entities
            entities = self.entity_extractor.extract_entities(content, metadata)
            
            # Extract relationships
            relationships = self.relationship_extractor.extract_relationships(content, entities)
            
            # Store entities
            stored_entities = 0
            for entity in entities:
                if await self.add_entity(entity):
                    stored_entities += 1
            
            # Store relationships
            stored_relationships = 0
            for relationship in relationships:
                if await self.add_relationship(relationship):
                    stored_relationships += 1
            
            return {
                "entities_extracted": len(entities),
                "entities_stored": stored_entities,
                "relationships_extracted": len(relationships),
                "relationships_stored": stored_relationships,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.is_connected = False
            logger.info("Neo4j connection closed")

class KnowledgeGraphService:
    """High-level service for knowledge graph operations"""
    
    def __init__(self):
        self.graph = Neo4jKnowledgeGraph()
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize knowledge graph service"""
        try:
            await self.graph.initialize()
            self.is_initialized = True
            logger.info("Knowledge graph service initialized")
        except Exception as e:
            logger.error(f"Error initializing knowledge graph service: {e}")
            raise
    
    async def analyze_investment_relationships(self, company_id: str) -> Dict[str, Any]:
        """Analyze investment relationships for a company"""
        try:
            # Find related entities
            related = await self.graph.find_related_entities(
                company_id, 
                [RelationshipType.COMPETES_WITH, RelationshipType.SUPPLIES_TO, RelationshipType.OWNS],
                max_depth=2
            )
            
            # Categorize relationships
            competitors = []
            suppliers = []
            subsidiaries = []
            
            for entity, relationships in related:
                for rel in relationships:
                    if rel.relationship_type == RelationshipType.COMPETES_WITH:
                        competitors.append({
                            "entity": entity.name,
                            "strength": rel.strength,
                            "properties": rel.properties
                        })
                    elif rel.relationship_type == RelationshipType.SUPPLIES_TO:
                        suppliers.append({
                            "entity": entity.name,
                            "strength": rel.strength,
                            "properties": rel.properties
                        })
                    elif rel.relationship_type == RelationshipType.OWNS:
                        subsidiaries.append({
                            "entity": entity.name,
                            "strength": rel.strength,
                            "properties": rel.properties
                        })
            
            return {
                "company_id": company_id,
                "competitors": competitors,
                "suppliers": suppliers,
                "subsidiaries": subsidiaries,
                "total_relationships": len(competitors) + len(suppliers) + len(subsidiaries),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing investment relationships: {e}")
            return {"error": str(e)}
    
    async def find_investment_opportunities(self, sector: str, criteria: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Find investment opportunities based on knowledge graph analysis"""
        try:
            # Search for companies in sector
            companies = await self.graph.search_entities(
                sector, 
                [EntityType.COMPANY], 
                limit=50
            )
            
            opportunities = []
            for company in companies:
                # Analyze company relationships
                analysis = await self.analyze_investment_relationships(company.id)
                
                # Score opportunity based on relationships
                opportunity_score = 0
                
                # More competitors might indicate a competitive market
                competitor_count = len(analysis.get("competitors", []))
                if competitor_count > 0:
                    opportunity_score += min(competitor_count * 0.1, 0.5)
                
                # Suppliers indicate business ecosystem
                supplier_count = len(analysis.get("suppliers", []))
                if supplier_count > 0:
                    opportunity_score += min(supplier_count * 0.15, 0.3)
                
                # Subsidiaries indicate diversification
                subsidiary_count = len(analysis.get("subsidiaries", []))
                if subsidiary_count > 0:
                    opportunity_score += min(subsidiary_count * 0.2, 0.4)
                
                opportunities.append({
                    "company": company.name,
                    "company_id": company.id,
                    "opportunity_score": opportunity_score,
                    "analysis": analysis,
                    "properties": company.properties
                })
            
            # Sort by opportunity score
            opportunities.sort(key=lambda x: x["opportunity_score"], reverse=True)
            
            return opportunities[:10]  # Return top 10
            
        except Exception as e:
            logger.error(f"Error finding investment opportunities: {e}")
            return []
    
    async def get_market_insights(self) -> Dict[str, Any]:
        """Get market insights from knowledge graph"""
        try:
            stats = await self.graph.get_graph_statistics()
            
            # Find most connected entities
            most_connected = await self._find_most_connected_entities()
            
            # Analyze sector distribution
            sector_analysis = await self._analyze_sector_distribution()
            
            return {
                "graph_statistics": stats,
                "most_connected_entities": most_connected,
                "sector_analysis": sector_analysis,
                "insights_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting market insights: {e}")
            return {"error": str(e)}
    
    async def _find_most_connected_entities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Find most connected entities in the graph"""
        if not self.graph.is_connected:
            return []
        
        try:
            with self.graph.driver.session() as session:
                query = """
                MATCH (e:Entity)-[r]-()
                RETURN e.id as id, e.name as name, e.entity_type as type, count(r) as connections
                ORDER BY connections DESC
                LIMIT $limit
                """
                
                result = session.run(query, {"limit": limit})
                
                return [{
                    "id": record["id"],
                    "name": record["name"],
                    "type": record["type"],
                    "connections": record["connections"]
                } for record in result]
                
        except Exception as e:
            logger.error(f"Error finding most connected entities: {e}")
            return []
    
    async def _analyze_sector_distribution(self) -> Dict[str, Any]:
        """Analyze distribution of entities across sectors"""
        if not self.graph.is_connected:
            return {}
        
        try:
            with self.graph.driver.session() as session:
                query = """
                MATCH (s:Entity {entity_type: 'Sector'})-[r]-(c:Entity {entity_type: 'Company'})
                RETURN s.name as sector, count(c) as companies
                ORDER BY companies DESC
                """
                
                result = session.run(query)
                
                sector_data = {}
                for record in result:
                    sector_data[record["sector"]] = record["companies"]
                
                return {
                    "sector_distribution": sector_data,
                    "total_sectors": len(sector_data),
                    "most_populated_sector": max(sector_data.items(), key=lambda x: x[1]) if sector_data else None
                }
                
        except Exception as e:
            logger.error(f"Error analyzing sector distribution: {e}")
            return {}
    
    async def cleanup(self):
        """Cleanup knowledge graph service"""
        await self.graph.close()
        logger.info("Knowledge graph service cleaned up")

# Factory function
def create_knowledge_graph_service() -> KnowledgeGraphService:
    """Create and return knowledge graph service"""
    return KnowledgeGraphService()